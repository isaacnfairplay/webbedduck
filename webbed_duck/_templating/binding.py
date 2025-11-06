"""Helpers for validating and binding route parameters."""

from __future__ import annotations

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
    ) -> None:
        self._parameters: Dict[str, ResolvedParameter] = dict(parameters)
        self._template_consumed: set[str] = set()
        self._template_view: Mapping[str, Any] | None = None

    def __getitem__(self, key: str) -> ResolvedParameter:
        return self._parameters[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

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

            guard_errors = _run_guards(name, value, spec.guards)
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


def _run_guards(name: str, value: Any, guards: Mapping[str, Any]) -> list[str]:
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

    return errors
