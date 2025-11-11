"""Helpers for validating and binding route parameters."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping

from .errors import TemplateApplicationError
from .guards import validate_parameter_guards
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
            raw_result = _resolve_raw_parameter(
                name, spec, provided, consumed_keys, errors
            )
            if raw_result is None:
                continue
            raw_value, provenance = raw_result

            value = _coerce_parameter_value(name, spec, raw_value, errors)
            if value is None:
                continue

            if not _validate_parameter_guards(name, value, spec, resolved, errors):
                continue

            resolved[name] = ResolvedParameter(
                name=name,
                value=value,
                provenance=provenance,
                allow_template=spec.allow_template,
                allow_binding=spec.allow_binding,
                spec=spec,
            )

        unknown_parameters, unknown_errors = _collect_unknown_parameters(
            provided, consumed_keys, context.allow_unknown_parameters
        )
        errors.extend(unknown_errors)

        if errors:
            raise ParameterBindingError("; ".join(errors))

        resolved.update(
            {
                key: ResolvedParameter(
                    name=key,
                    value=value,
                    provenance="provided",
                    allow_template=False,
                    allow_binding=True,
                    spec=None,
                )
                for key, value in unknown_parameters.items()
            }
        )

        return cls(resolved)

    def for_template(self) -> Mapping[str, Any]:
        if self._template_view is None:
            data: Dict[str, Any] = {
                name: resolved.value
                for name, resolved in self._parameters.items()
                if resolved.allow_template
            }

            self._template_view = _TemplateParameterNamespace(self, data)
        return self._template_view

    def _binding_candidates(self) -> Dict[str, Any]:
        return {
            name: parameter.value
            for name, parameter in self._parameters.items()
            if parameter.allow_binding and name not in self._template_consumed
        }

    def for_binding(self, *, used_names: Iterable[str] | None = None) -> Dict[str, Any]:
        candidates = self._binding_candidates()
        if used_names is None:
            return candidates
        allowed = set(used_names)
        return {name: value for name, value in candidates.items() if name in allowed}

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


def _resolve_raw_parameter(
    name: str,
    spec: ParameterSpec,
    provided: Mapping[str, Any],
    consumed_keys: set[str],
    errors: list[str],
) -> tuple[Any, str] | None:
    raw_value = provided.get(name, _MISSING)
    if raw_value is not _MISSING:
        consumed_keys.add(name)
        return raw_value, "provided"
    if spec.default is not _MISSING:
        return spec.default, "default"
    if spec.required:
        errors.append(f"Parameter '{name}' is required")
    return None


def _coerce_parameter_value(
    name: str,
    spec: ParameterSpec,
    raw_value: Any,
    errors: list[str],
) -> Any | None:
    try:
        return _coerce_value(name, spec.type, raw_value)
    except ParameterBindingError as exc:
        errors.append(str(exc))
        return None


def _validate_parameter_guards(
    name: str,
    value: Any,
    spec: ParameterSpec,
    resolved: Mapping[str, "ResolvedParameter"],
    errors: list[str],
) -> bool:
    guard_errors = validate_parameter_guards(name, value, spec.guards, resolved)
    if guard_errors:
        errors.extend(guard_errors)
        return False
    return True


_BOOLEAN_STRINGS = {"true": True, "1": True, "yes": True, "on": True, "false": False, "0": False, "no": False, "off": False}


def _coerce_boolean(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
    elif isinstance(value, str):
        normalized = _BOOLEAN_STRINGS.get(value.strip().lower())
        if normalized is not None:
            return normalized
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


_TYPE_COERCERS: Mapping[str, Callable[[str, Any], Any]] = {
    "string": lambda _name, value: value if isinstance(value, str) else str(value), "boolean": _coerce_boolean,
    "integer": _numeric_coercer(int, "an integer"), "number": _numeric_coercer(float, "numeric"),
}


def _coerce_value(name: str, type_name: str, value: Any) -> Any:
    coercer = _TYPE_COERCERS.get(type_name)
    if coercer is None:
        raise ParameterBindingError(
            f"Parameter '{name}' uses unsupported type '{type_name}'"
        )
    return coercer(name, value)





