"""Helpers for validating and binding route parameters."""

from __future__ import annotations

import copy
import datetime as _dt
import operator
import re
from collections.abc import Mapping as MappingABC
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping

from .errors import TemplateApplicationError
from .state import StringNamespace

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
            raise TemplateApplicationError("Parameter configuration must be a mapping")
        self._configuration = copy.deepcopy(configuration)

    def with_configuration(
        self, configuration: Mapping[str, Any]
    ) -> "ParameterContext":
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
            ready, raw_value, provenance = cls._resolve_parameter_source(
                name, spec, provided, consumed_keys, errors
            )
            if not ready:
                continue

            try:
                value = _coerce_value(name, spec.type, raw_value)
            except ParameterBindingError as exc:
                errors.append(str(exc))
                continue

            guard_errors = _run_guards(name, value, spec.guards, resolved)
            if guard_errors:
                errors.extend(guard_errors)
                continue

            # fmt: off
            resolved[name] = ResolvedParameter(name, value, provenance, spec.allow_template, spec.allow_binding, spec)
            # fmt: on

        unknown_parameters, unknown_errors = _collect_unknown_parameters(
            provided, consumed_keys, context.allow_unknown_parameters
        )
        errors.extend(unknown_errors)

        # fmt: off
        if errors: raise ParameterBindingError("; ".join(errors))  # noqa: E701
        # fmt: on

        # fmt: off
        resolved.update({key: ResolvedParameter(key, value, "provided", False, True, None) for key, value in unknown_parameters.items()})
        # fmt: on

        return cls(resolved)

    @staticmethod
    def _resolve_parameter_source(
        name: str,
        spec: ParameterSpec,
        provided: Mapping[str, Any],
        consumed: set[str],
        errors: list[str],
    ) -> tuple[bool, Any, str]:
        raw_value = provided.get(name, _MISSING)
        if raw_value is _MISSING:
            if spec.default is _MISSING:
                if spec.required:
                    errors.append(f"Parameter '{name}' is required")
                return False, _MISSING, "provided"
            return True, spec.default, "default"

        consumed.add(name)
        return True, raw_value, "provided"

    def for_template(self) -> Mapping[str, Any]:
        if self._template_view is None:
            data: Dict[str, Any] = {
                name: resolved.value
                for name, resolved in self._parameters.items()
                if resolved.allow_template
            }

            self._template_view = _TemplateParameterNamespace(self, data)
        return self._template_view

    def for_binding(self, *, used_names: Iterable[str] | None = None) -> Dict[str, Any]:
        names = None if used_names is None else set(used_names)
        return {
            name: parameter.value
            for name, parameter in self._parameters.items()
            if parameter.allow_binding
            and name not in self._template_consumed
            and (names is None or name in names)
        }

    @property
    def template_consumed(self) -> set[str]:
        return set(self._template_consumed)


class _TemplateParameterNamespace(StringNamespace):
    __slots__ = ("_context",)

    def __init__(self, context: "ParameterContext", data: Mapping[str, Any]) -> None:
        super().__init__(data, data.keys())
        self._context = context

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        value = super().__getitem__(key)
        self._context._template_consumed.add(key)
        return value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return self[key] if key in self else default


def _collect_unknown_parameters(
    provided: Mapping[str, Any],
    consumed_keys: set[str],
    allow_unknown: bool,
) -> tuple[dict[str, Any], list[str]]:
    if not (unknown_keys := set(provided) - consumed_keys):
        return {}, []
    if not allow_unknown:
        ordered = sorted(unknown_keys)
        if len(ordered) == 1:
            return {}, [f"Unknown parameter '{ordered[0]}'"]
        return {}, ["Unknown parameters: " + ", ".join(ordered)]
    return {key: provided[key] for key in unknown_keys}, []


# fmt: off
_BOOLEAN_STRINGS = {"true": True, "1": True, "yes": True, "on": True, "false": False, "0": False, "no": False, "off": False}
# fmt: on


def _coerce_boolean(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        try:
            return _BOOLEAN_STRINGS[value.strip().lower()]
        except KeyError:
            pass
    raise ParameterBindingError(f"Parameter '{name}' must be a boolean")


def _numeric_coercer(
    func: Callable[[Any], Any], description: str
) -> Callable[[str, Any], Any]:
    def _inner(name: str, value: Any) -> Any:
        try:
            return func(value)
        except (TypeError, ValueError) as exc:
            raise ParameterBindingError(
                f"Parameter '{name}' must be {description}"
            ) from exc

    return _inner


# fmt: off
_TYPE_COERCERS: Mapping[str, Callable[[str, Any], Any]] = {
    "string": lambda _name, value: value if isinstance(value, str) else str(value), "boolean": _coerce_boolean,
    "integer": _numeric_coercer(int, "an integer"), "number": _numeric_coercer(float, "numeric"),
}
# fmt: on


def _coerce_value(name: str, type_name: str, value: Any) -> Any:
    coercer = _TYPE_COERCERS.get(type_name)
    if coercer is None:
        raise ParameterBindingError(
            f"Parameter '{name}' uses unsupported type '{type_name}'"
        )
    return coercer(name, value)


_GuardPredicate = Callable[[Any], bool] | type | tuple[type, ...]


def _guard_option(
    name: str,
    guards: Mapping[str, Any],
    key: str,
    *,
    predicate: _GuardPredicate | None = None,
    type_label: str,
    transform: Callable[[Any], Any] | None = None,
) -> tuple[Any | None, list[str]]:
    value = guards.get(key)
    if value is None:
        return None, []
    if predicate is not None:
        check = (
            isinstance(value, predicate)
            if isinstance(predicate, (tuple, type))
            else predicate(value)
        )
        if not check:
            return None, [f"Parameter '{name}' {key} guard must be {type_label}"]
    return (transform(value) if transform else value), []


# fmt: off
_SIMPLE_GUARDS: Mapping[
    str,
    tuple[
        _GuardPredicate | None,
        str,
        Callable[[Any], Any] | None,
        Callable[[str, Any, Any], Iterable[str]],
    ],
] = {
    "choices": (
        lambda candidate: isinstance(candidate, Iterable)
        and not isinstance(candidate, (str, bytes)),
        "iterable",
        list,
        lambda name, current, options: []
        if current in options
        else [f"Parameter '{name}' must be one of: {', '.join(map(str, options))}"],
    ),
    "regex": (
        str,
        "a string pattern",
        None,
        lambda name, current, pattern: (
            [f"Parameter '{name}' must be a string to apply regex guard"]
            if not isinstance(current, str)
            else (
                []
                if re.fullmatch(pattern, current)
                else [f"Parameter '{name}' must match pattern '{pattern}'"]
            )
        ),
    ),
}
# fmt: on


def _iter_simple_guard_errors(
    name: str, value: Any, guards: Mapping[str, Any]
) -> Iterable[str]:
    for key, (predicate, type_label, transform, validator) in _SIMPLE_GUARDS.items():
        option, type_errors = _guard_option(
            name,
            guards,
            key,
            predicate=predicate,
            type_label=type_label,
            transform=transform,
        )
        if type_errors:
            yield from type_errors
        elif option is not None:
            yield from validator(name, value, option)


def _run_guards(
    name: str,
    value: Any,
    guards: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    if not guards:
        return []

    errors = list(_iter_simple_guard_errors(name, value, guards))
    for key, handler in _MAPPING_GUARDS.items():
        mapping, type_errors = _guard_option(
            name,
            guards,
            key,
            predicate=MappingABC,
            type_label="a mapping",
        )
        if type_errors:
            errors.extend(type_errors)
        elif mapping is not None:
            errors.extend(handler(name, value, mapping, resolved))
    return errors


# fmt: off
_BOUND_CHECKS: Mapping[str, tuple[Callable[[Any, Any], bool], str]] = {"min": (operator.lt, "at least"), "max": (operator.gt, "at most")}
# fmt: on


def _coerce_guard_values(
    name: str,
    guard_key: str,
    bounds: Mapping[str, Any],
    converter: Callable[[Any], float | int | None],
) -> tuple[dict[str, tuple[Any, float | int]], list[str]]:
    numeric: dict[str, tuple[Any, float | int]] = {}
    errors: list[str] = []
    for label in ("min", "max"):
        raw = bounds.get(label)
        if raw is None:
            continue
        coerced = converter(raw)
        if coerced is None:
            errors.append(
                f"Parameter '{name}' {guard_key} guard '{label}' must be numeric"
            )
        else:
            numeric[label] = (raw, coerced)
    return numeric, errors


def _validate_length_guard(
    name: str,
    value: Any,
    bounds: Mapping[str, Any],
    _resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    try:
        length = len(value)  # type: ignore[arg-type]
    except TypeError:
        return [f"Parameter '{name}' length guard requires a sized value"]

    numeric, errors = _coerce_guard_values(name, "length", bounds, _as_int)
    return errors or [
        f"Parameter '{name}' length must be {_BOUND_CHECKS[label][1]} {raw}"
        for label, (raw, coerced) in numeric.items()
        if _BOUND_CHECKS[label][0](length, coerced)
    ]


# fmt: off
def _range_violation_messages(name: str, numeric_value: float | int, numeric: Mapping[str, tuple[Any, float | int]]) -> list[str]:
# fmt: on
    active = [
        (label, raw)
        for label, (raw, coerced) in numeric.items()
        if _BOUND_CHECKS[label][0](numeric_value, coerced)
    ]
    if not active:
        return []
    if {"min", "max"}.issubset(numeric):
        min_raw, _ = numeric["min"]
        max_raw, _ = numeric["max"]
        return [f"Parameter '{name}' must be between {min_raw} and {max_raw}"]
    label, raw = active[0]
    return [f"Parameter '{name}' must be {_BOUND_CHECKS[label][1]} {raw}"]


def _validate_range_guard(
    name: str,
    value: Any,
    bounds: Mapping[str, Any],
    _resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    if not any(bounds.get(label) is not None for label in ("min", "max")):
        return [f"Parameter '{name}' range guard requires 'min' or 'max'"]
    numeric_value = _as_float(value)
    if numeric_value is None:
        return [f"Parameter '{name}' must be numeric to apply range guard"]

    numeric, errors = _coerce_guard_values(name, "range", bounds, _as_float)
    return errors or _range_violation_messages(name, numeric_value, numeric)


def _validate_datetime_window_guard(
    name: str,
    value: Any,
    window: Mapping[str, Any],
    _resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    earliest = window.get("earliest")
    latest = window.get("latest")
    if earliest is None and latest is None:
        return [
            f"Parameter '{name}' datetime_window guard requires 'earliest' or 'latest'",
        ]

    value_dt, value_error = _parse_datetime_for_guard(value, name, for_value=True)
    if value_error is not None:
        return [value_error]

    def _check_boundary(
        label: str,
        raw_boundary: Any,
        comparator: Callable[[Any, Any], bool],
        descriptor: str,
    ) -> str | None:
        if raw_boundary is None:
            return None
        boundary_dt, boundary_error = _parse_datetime_for_guard(
            raw_boundary, name, boundary=label
        )
        if boundary_error is not None:
            return boundary_error
        if value_dt is None or boundary_dt is None:
            return None
        if _datetimes_mixed_timezone_awareness(value_dt, boundary_dt):
            return f"Parameter '{name}' datetime_window guard requires value and {label} boundary to use the same timezone awareness"
        if comparator(value_dt, boundary_dt):
            formatted = raw_boundary.isoformat() if isinstance(raw_boundary, _dt.datetime) else str(raw_boundary)
            return f"Parameter '{name}' must not be {descriptor} {formatted}"
        return None

    return [
        error
        for error in (
            _check_boundary("earliest", earliest, operator.lt, "earlier than"),
            _check_boundary("latest", latest, operator.gt, "later than"),
        )
        if error is not None
    ]


def _resolve_compare_configuration(
    name: str,
    compare: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> tuple[str | None, "_CompareOperator" | None, "ResolvedParameter" | None, list[str]]:
    target_name = compare.get("parameter")
    if not isinstance(target_name, str):
        return (
            None,
            None,
            None,
            [f"Parameter '{name}' compare guard requires a parameter name"],
        )

    operator_name = compare.get("operator") or "eq"
    if not isinstance(operator_name, str):
        return (
            target_name,
            None,
            None,
            [f"Parameter '{name}' compare guard operator must be a string"],
        )

    comparator = _COMPARE_OPERATORS.get(operator_name)
    if comparator is None:
        return (
            target_name,
            None,
            None,
            [
                f"Parameter '{name}' compare guard uses unsupported operator '{operator_name}'"
            ],
        )

    other = resolved.get(target_name)
    if other is None:
        return (
            target_name,
            comparator,
            None,
            [
                f"Parameter '{name}' compare guard requires parameter '{target_name}' to be resolved first"
            ],
        )

    return target_name, comparator, other, []


def _validate_compare_guard(
    name: str,
    value: Any,
    compare: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    target_name, comparator, other, config_errors = _resolve_compare_configuration(
        name, compare, resolved
    )
    if config_errors:
        return config_errors

    with suppress(TypeError):
        assert comparator is not None and other is not None
        if comparator.compare(value, other.value):
            return []

        message = compare.get("message")
        if isinstance(message, str):
            return [message]
        if message is None:
            return [comparator.message(name, target_name)]
        return [str(message)]

    return [
        f"Parameter '{name}' compare guard could not compare with parameter '{target_name}'",
    ]


# fmt: off
_MAPPING_GUARDS: Mapping[
    str,
    Callable[[str, Any, Mapping[str, Any], Mapping[str, "ResolvedParameter"]], list[str]],
] = {
    "length": _validate_length_guard,
    "range": _validate_range_guard,
    "datetime_window": _validate_datetime_window_guard,
    "compare": _validate_compare_guard,
}
# fmt: on


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
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


def _datetimes_mixed_timezone_awareness(
    left: _dt.datetime | None, right: _dt.datetime | None
) -> bool:
    if left is None or right is None:
        return False
    return (left.tzinfo is None) != (right.tzinfo is None)


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
        return f"Parameter '{name}' must be {self._description} parameter '{other}'"


_COMPARE_OPERATORS: Mapping[str, _CompareOperator] = {
    "eq": _CompareOperator(operator.eq, "equal to"),
    "ne": _CompareOperator(operator.ne, "different from"),
    "lt": _CompareOperator(operator.lt, "less than"),
    "lte": _CompareOperator(operator.le, "less than or equal to"),
    "gt": _CompareOperator(operator.gt, "greater than"),
    "gte": _CompareOperator(operator.ge, "greater than or equal to"),
}
