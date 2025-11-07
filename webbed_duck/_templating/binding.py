"""Helpers for validating and binding route parameters."""

from __future__ import annotations

import copy
import datetime as _dt
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping

from .errors import TemplateApplicationError

__all__ = [
    "ParameterBindingError",
    "ParameterSpec",
    "ValidationContext",
    "ResolvedParameter",
    "ParameterContext",
]


_MISSING = object()


class ParameterBindingError(TemplateApplicationError):
    """Raised when request parameters fail validation."""

    __slots__ = ()


@dataclass(frozen=True)
class ParameterSpec:
    """Immutable description of a single parameter."""

    name: str
    type: str
    required: bool = False
    default: Any = _MISSING
    allow_template: bool = False
    allow_binding: bool = True
    guards: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_manifest(cls, name: str, data: Mapping[str, Any]) -> "ParameterSpec":
        parameter_type = data.get("type")
        if not isinstance(parameter_type, str):
            raise ParameterBindingError(
                f"Parameter '{name}' must declare a string type"
            )

        required = bool(data.get("required", False))
        allow_template = bool(data.get("allow_template", False))
        allow_binding = bool(data.get("allow_binding", True))
        default = data.get("default", _MISSING)
        guards = data.get("guards", {})
        if guards is None:
            guards = {}
        if not isinstance(guards, Mapping):
            raise ParameterBindingError(
                f"Parameter '{name}' guards must be a mapping if provided"
            )

        return cls(
            name=name,
            type=parameter_type,
            required=required,
            default=default,
            allow_template=allow_template,
            allow_binding=allow_binding,
            guards=dict(guards),
        )


@dataclass(frozen=True)
class ValidationContext:
    """Container storing parsed parameter specifications."""

    specs: Mapping[str, ParameterSpec]
    allow_unknown_parameters: bool = False

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "ValidationContext":
        parameters = manifest.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ParameterBindingError("Parameter manifest must be a mapping")

        specs: Dict[str, ParameterSpec] = {}
        for name, data in parameters.items():
            if not isinstance(data, Mapping):
                raise ParameterBindingError(
                    f"Configuration for parameter '{name}' must be a mapping"
                )
            specs[name] = ParameterSpec.from_manifest(name, data)

        allow_unknown = bool(manifest.get("allow_unknown_parameters", False))
        return cls(specs=specs, allow_unknown_parameters=allow_unknown)

    def resolve(self, provided: Mapping[str, Any]) -> "ParameterContext":
        return ParameterContext.from_manifest(self, provided)


@dataclass(frozen=True)
class ResolvedParameter:
    """Concrete value for a parameter after validation."""

    name: str
    value: Any
    provenance: str
    allow_template: bool
    allow_binding: bool
    spec: ParameterSpec | None


class ParameterContext(Mapping[str, ResolvedParameter]):
    """Read-only view over validated parameters."""

    def __init__(
        self,
        parameters: Mapping[str, ResolvedParameter],
        *,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self._parameters: Dict[str, ResolvedParameter] = dict(parameters)
        self._template_consumed: set[str] = set()
        self._template_view: Mapping[str, Any] | None = None
        self._configuration: Mapping[str, Any] = {}
        if configuration is not None:
            self.attach_configuration(configuration)

    def __getitem__(self, key: str) -> ResolvedParameter:
        return self._parameters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    @property
    def configuration(self) -> Mapping[str, Any]:
        """Return the stored template configuration for the context."""

        return self._configuration

    def attach_configuration(self, configuration: Mapping[str, Any]) -> None:
        """Store ``configuration`` for later use during context preparation."""

        if not isinstance(configuration, Mapping):
            raise TemplateApplicationError(
                "Parameter configuration must be a mapping"
            )
        self._configuration = copy.deepcopy(configuration)

    def with_configuration(self, configuration: Mapping[str, Any]) -> "ParameterContext":
        """Attach ``configuration`` and return ``self`` for chaining."""

        self.attach_configuration(configuration)
        return self

    @classmethod
    def from_manifest(
        cls,
        context: ValidationContext,
        provided: Mapping[str, Any],
    ) -> "ParameterContext":
        resolved: Dict[str, ResolvedParameter] = {}
        errors: list[str] = []
        consumed_keys: set[str] = set()

        for name, spec in context.specs.items():
            raw_value = provided.get(name, _MISSING)
            provenance = "provided"
            if raw_value is _MISSING:
                if spec.default is not _MISSING:
                    raw_value = spec.default
                    provenance = "default"
                elif spec.required:
                    errors.append(f"Parameter '{name}' is required")
                    continue
                else:
                    continue
            else:
                consumed_keys.add(name)

            try:
                value = _coerce_value(name, spec.type, raw_value)
            except ParameterBindingError as exc:
                errors.append(str(exc))
                continue

            guard_errors = _run_guards(name, value, spec.guards, resolved)
            if guard_errors:
                errors.extend(guard_errors)
                continue

            resolved[name] = ResolvedParameter(
                name=name,
                value=value,
                provenance=provenance,
                allow_template=spec.allow_template,
                allow_binding=spec.allow_binding,
                spec=spec,
            )

        unknown_keys = set(provided.keys()) - consumed_keys
        unknown_parameters: Dict[str, Any] = {}
        if unknown_keys:
            if not context.allow_unknown_parameters:
                ordered = sorted(unknown_keys)
                if len(ordered) == 1:
                    errors.append(f"Unknown parameter '{ordered[0]}'")
                else:
                    errors.append(
                        "Unknown parameters: " + ", ".join(ordered)
                    )
            else:
                unknown_parameters = {key: provided[key] for key in unknown_keys}

        if errors:
            raise ParameterBindingError("; ".join(errors))

        for key, value in unknown_parameters.items():
            resolved[key] = ResolvedParameter(
                name=key,
                value=value,
                provenance="provided",
                allow_template=False,
                allow_binding=True,
                spec=None,
            )

        return cls(resolved)

    def for_template(self) -> Mapping[str, Any]:
        if self._template_view is None:
            from .state import StringNamespace

            data: Dict[str, Any] = {
                name: resolved.value
                for name, resolved in self._parameters.items()
                if resolved.allow_template
            }

            context = self

            class TemplateParameterNamespace(StringNamespace):
                def __getitem__(self, key: str) -> Any:  # type: ignore[override]
                    value = super().__getitem__(key)
                    context._template_consumed.add(key)
                    return value

                def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
                    if key in self:
                        return self[key]
                    return default

                def __contains__(self, key: object) -> bool:  # type: ignore[override]
                    return dict.__contains__(self, key)

            self._template_view = TemplateParameterNamespace(data, data.keys())
        return self._template_view

    def for_binding(self, *, used_names: Iterable[str] | None = None) -> Dict[str, Any]:
        names = None if used_names is None else set(used_names)
        binding: Dict[str, Any] = {}
        for name, parameter in self._parameters.items():
            if not parameter.allow_binding:
                continue
            if name in self._template_consumed:
                continue
            if names is not None and name not in names:
                continue
            binding[name] = parameter.value
        return binding

    @property
    def template_consumed(self) -> set[str]:
        return set(self._template_consumed)


def _coerce_value(name: str, type_name: str, value: Any) -> Any:
    if type_name == "string":
        if isinstance(value, str):
            return value
        return str(value)
    if type_name == "integer":
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError(
                f"Parameter '{name}' must be an integer"
            ) from exc
    if type_name == "number":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError(
                f"Parameter '{name}' must be numeric"
            ) from exc
    if type_name == "boolean":
        try:
            return _coerce_boolean(value)
        except ValueError as exc:
            raise ParameterBindingError(
                f"Parameter '{name}' must be a boolean"
            ) from exc
    raise ParameterBindingError(
        f"Parameter '{name}' uses unsupported type '{type_name}'"
    )


def _coerce_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError("Boolean numeric coercion expects 0 or 1")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError("Cannot coerce value to boolean")


def _run_guards(
    name: str,
    value: Any,
    guards: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    errors: list[str] = []
    if not guards:
        return errors

    if "choices" in guards:
        choices = guards["choices"]
        if isinstance(choices, Iterable) and not isinstance(choices, str):
            if value not in choices:
                joined = ", ".join(map(str, choices))
                errors.append(f"Parameter '{name}' must be one of: {joined}")
        else:
            errors.append(
                f"Parameter '{name}' choices guard must be iterable"
            )

    if "regex" in guards:
        pattern = guards["regex"]
        if not isinstance(pattern, str):
            errors.append(
                f"Parameter '{name}' regex guard must be a string pattern"
            )
        else:
            if not isinstance(value, str):
                errors.append(
                    f"Parameter '{name}' must be a string to apply regex guard"
                )
            elif re.fullmatch(pattern, value) is None:
                errors.append(
                    f"Parameter '{name}' must match pattern '{pattern}'"
                )

    if "length" in guards:
        bounds = guards["length"]
        if not isinstance(bounds, Mapping):
            errors.append(
                f"Parameter '{name}' length guard must be a mapping"
            )
        else:
            minimum = bounds.get("min")
            maximum = bounds.get("max")
            try:
                length = len(value)  # type: ignore[arg-type]
            except TypeError:
                errors.append(
                    f"Parameter '{name}' length guard requires a sized value"
                )
            else:
                if minimum is not None and length < int(minimum):
                    errors.append(
                        f"Parameter '{name}' length must be at least {minimum}"
                    )
                if maximum is not None and length > int(maximum):
                    errors.append(
                        f"Parameter '{name}' length must be at most {maximum}"
                    )

    if "range" in guards:
        bounds = guards["range"]
        if not isinstance(bounds, Mapping):
            errors.append(
                f"Parameter '{name}' range guard must be a mapping"
            )
        else:
            minimum = bounds.get("min")
            maximum = bounds.get("max")
            if minimum is None and maximum is None:
                errors.append(
                    f"Parameter '{name}' range guard requires 'min' or 'max'"
                )
            else:
                numeric_value = _as_float(value)
                if numeric_value is None:
                    errors.append(
                        f"Parameter '{name}' must be numeric to apply range guard"
                    )
                else:
                    min_value = _as_float(minimum) if minimum is not None else None
                    max_value = _as_float(maximum) if maximum is not None else None
                    range_error: str | None = None
                    range_setup_errors: list[str] = []
                    if minimum is not None and min_value is None:
                        range_setup_errors.append(
                            f"Parameter '{name}' range guard 'min' must be numeric"
                        )
                    if maximum is not None and max_value is None:
                        range_setup_errors.append(
                            f"Parameter '{name}' range guard 'max' must be numeric"
                        )
                    if range_setup_errors:
                        errors.extend(range_setup_errors)
                    elif min_value is not None and numeric_value < min_value:
                        if max_value is not None:
                            range_error = (
                                f"Parameter '{name}' must be between {minimum} and {maximum}"
                            )
                        else:
                            range_error = (
                                f"Parameter '{name}' must be at least {minimum}"
                            )
                    if (
                        range_error is None
                        and max_value is not None
                        and numeric_value > max_value
                    ):
                        if min_value is not None:
                            range_error = (
                                f"Parameter '{name}' must be between {minimum} and {maximum}"
                            )
                        else:
                            range_error = (
                                f"Parameter '{name}' must be at most {maximum}"
                            )
                    if range_error is not None:
                        errors.append(range_error)

    if "datetime_window" in guards:
        window = guards["datetime_window"]
        if not isinstance(window, Mapping):
            errors.append(
                f"Parameter '{name}' datetime_window guard must be a mapping"
            )
        else:
            earliest = window.get("earliest")
            latest = window.get("latest")
            if earliest is None and latest is None:
                errors.append(
                    f"Parameter '{name}' datetime_window guard requires 'earliest' or 'latest'"
                )
            else:
                value_dt, value_error = _parse_datetime_for_guard(
                    value, name, for_value=True
                )
                if value_error is not None:
                    errors.append(value_error)
                else:
                    if earliest is not None:
                        earliest_dt, earliest_error = _parse_datetime_for_guard(
                            earliest, name, boundary="earliest"
                        )
                        if earliest_error is not None:
                            errors.append(earliest_error)
                        elif value_dt is not None and value_dt < earliest_dt:
                            errors.append(
                                f"Parameter '{name}' must not be earlier than {_format_datetime_boundary(earliest)}"
                            )
                    if latest is not None:
                        latest_dt, latest_error = _parse_datetime_for_guard(
                            latest, name, boundary="latest"
                        )
                        if latest_error is not None:
                            errors.append(latest_error)
                        elif value_dt is not None and value_dt > latest_dt:
                            errors.append(
                                f"Parameter '{name}' must not be later than {_format_datetime_boundary(latest)}"
                            )

    if "compare" in guards:
        compare = guards["compare"]
        if not isinstance(compare, Mapping):
            errors.append(
                f"Parameter '{name}' compare guard must be a mapping"
            )
        else:
            target_name = compare.get("parameter")
            operator_name = compare.get("operator", "eq")
            custom_message = compare.get("message")

            if not isinstance(target_name, str):
                errors.append(
                    f"Parameter '{name}' compare guard requires a parameter name"
                )
            else:
                if operator_name is None:
                    operator_key = "eq"
                elif isinstance(operator_name, str):
                    operator_key = operator_name
                else:
                    errors.append(
                        f"Parameter '{name}' compare guard operator must be a string"
                    )
                    operator_key = None
                comparator = (
                    _COMPARE_OPERATORS.get(operator_key) if operator_key is not None else None
                )
                if comparator is None:
                    errors.append(
                        f"Parameter '{name}' compare guard uses unsupported operator '{operator_name}'"
                    )
                else:
                    other = resolved.get(target_name)
                    if other is None:
                        errors.append(
                            f"Parameter '{name}' compare guard requires parameter '{target_name}' to be resolved first"
                        )
                    else:
                        try:
                            result = comparator.compare(value, other.value)
                        except TypeError:
                            errors.append(
                                f"Parameter '{name}' compare guard could not compare with parameter '{target_name}'"
                            )
                        else:
                            if not result:
                                if isinstance(custom_message, str):
                                    message = custom_message
                                elif custom_message is None:
                                    message = comparator.message(name, target_name)
                                else:
                                    message = str(custom_message)
                                errors.append(message)

    return errors


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime_for_guard(
    raw: Any, name: str, *, for_value: bool = False, boundary: str | None = None
) -> tuple[_dt.datetime | None, str | None]:
    if raw is None:
        return None, None
    if isinstance(raw, _dt.datetime):
        return raw, None
    if isinstance(raw, str):
        try:
            return _dt.datetime.fromisoformat(raw), None
        except ValueError:
            pass
    if for_value:
        return (
            None,
            f"Parameter '{name}' must be a datetime or ISO 8601 string to apply datetime_window guard",
        )
    qualifier = f" {boundary}" if boundary else ""
    return (
        None,
        f"Parameter '{name}' datetime_window{qualifier} must be a datetime or ISO 8601 string",
    )


def _format_datetime_boundary(raw: Any) -> str:
    if isinstance(raw, _dt.datetime):
        return raw.isoformat()
    return str(raw)


class _CompareOperator:
    __slots__ = ("_func", "_description")

    def __init__(self, func, description: str) -> None:
        self._func = func
        self._description = description

    def compare(self, left: Any, right: Any) -> bool:
        return self._func(left, right)

    def message(self, name: str, other: str) -> str:
        if self._description == "different from":
            return f"Parameter '{name}' must be different from parameter '{other}'"
        return (
            f"Parameter '{name}' must be {self._description} parameter '{other}'"
        )


_COMPARE_OPERATORS: Mapping[str, _CompareOperator] = {
    "eq": _CompareOperator(operator.eq, "equal to"),
    "ne": _CompareOperator(operator.ne, "different from"),
    "lt": _CompareOperator(operator.lt, "less than"),
    "lte": _CompareOperator(operator.le, "less than or equal to"),
    "gt": _CompareOperator(operator.gt, "greater than"),
    "gte": _CompareOperator(operator.ge, "greater than or equal to"),
}
