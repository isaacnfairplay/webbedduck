"""Helpers for extracting inline metadata from SQL templates."""

from __future__ import annotations

import ast
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, TypeVar

from webbed_duck._templating.binding import ParameterSpec, ValidationContext
from webbed_duck._templating.parameters import ParameterWhitelist

from .cache import CacheConfig
from .cache_support import InvariantFilter

__all__ = [
    "TemplateMetadataError",
    "TemplateDirective",
    "TemplateMetadata",
    "InvariantDescriptor",
    "RouteDescription",
    "collect_template_metadata",
    "build_route_registry",
]


class TemplateMetadataError(ValueError):
    """Raised when inline route metadata cannot be parsed."""


@dataclass(frozen=True)
class TemplateDirective:
    """Descriptor representing an inline validator/filter declaration."""

    kind: str
    target: str
    name: str
    options: Mapping[str, Any] = field(default_factory=dict)
    source: str = "comment"
    line: int | None = None


@dataclass(frozen=True)
class TemplateMetadata:
    """Container summarising inline directives discovered for a template."""

    directives: tuple[TemplateDirective, ...] = ()

    @property
    def validators(self) -> tuple[TemplateDirective, ...]:
        """Return only validator directives for convenience."""

        return tuple(
            directive for directive in self.directives if directive.kind == "validator"
        )


@dataclass(frozen=True)
class InvariantDescriptor:
    """Frozen representation of an invariant filter configuration."""

    name: str
    column: str
    separator: str
    case_insensitive: bool

    @classmethod
    def from_filter(cls, name: str, config: InvariantFilter) -> "InvariantDescriptor":
        column = config.column or config.key
        return cls(
            name=name,
            column=column,
            separator=config.separator,
            case_insensitive=config.case_insensitive,
        )


@dataclass(frozen=True)
class RouteDescription:
    """Aggregated description for a route template."""

    slug: str
    template_path: Path
    metadata: TemplateMetadata
    invariants: Mapping[str, InvariantDescriptor]


_T = TypeVar("_T")


def collect_template_metadata(
    template: str,
    *,
    request_context: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | ValidationContext | None = None,
) -> TemplateMetadata:
    """Parse ``template`` and return discovered inline directives."""

    directives = list(_parse_inline_directives(template))
    directives.extend(_parse_comment_directives(template))
    directives.extend(_iter_whitelist_directives(request_context))
    directives.extend(_iter_validation_directives(validation))
    return TemplateMetadata(directives=tuple(directives))


def build_route_registry(
    template_root: Path,
    *,
    cache_config: CacheConfig,
    request_context: Mapping[str, Any] | None = None,
    validations: Mapping[str, Mapping[str, Any] | ValidationContext] | None = None,
) -> Mapping[str, RouteDescription]:
    """Scan ``template_root`` and build a map of route descriptions."""

    invariant_descriptors = _freeze_mapping(
        {
            name: InvariantDescriptor.from_filter(name, invariant)
            for name, invariant in cache_config.invariants.items()
        }
    )

    if not template_root.exists():
        return MappingProxyType({})

    routes: dict[str, RouteDescription] = {}
    for path in sorted(template_root.rglob("*")):
        if not path.is_file() or not _looks_like_sql_template(path):
            continue
        template_text = path.read_text(encoding="utf-8")
        slug = _slug_for_template(path, template_root)
        validation = validations.get(slug) if validations else None
        metadata = collect_template_metadata(
            template_text,
            request_context=request_context,
            validation=validation,
        )
        routes[slug] = RouteDescription(
            slug=slug,
            template_path=path,
            metadata=metadata,
            invariants=invariant_descriptors,
        )

    return MappingProxyType(routes)


_PLACEHOLDER = re.compile(r"{{\s*(.*?)\s*}}")


def _parse_inline_directives(template: str) -> Iterable[TemplateDirective]:
    for match in _PLACEHOLDER.finditer(template):
        expression = match.group(1).strip()
        if not expression.startswith("webbed_duck."):
            continue
        line_number = template.count("\n", 0, match.start()) + 1
        directive = _parse_inline_expression(expression, line_number)
        if directive is not None:
            yield directive


def _parse_inline_expression(expression: str, line: int) -> TemplateDirective | None:
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:  # pragma: no cover - surfaced in tests
        raise TemplateMetadataError(
            f"Invalid inline directive on line {line}: {expression!r}"
        ) from exc

    if not isinstance(node, ast.Call):
        return None

    func = node.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "webbed_duck"
    ):
        return None

    if any(keyword.arg is None for keyword in node.keywords):
        raise TemplateMetadataError(
            f"Directive on line {line} does not support **kwargs"
        )

    args = [_literal_eval(arg, line) for arg in node.args]
    kwargs: MutableMapping[str, Any] = {
        keyword.arg: _literal_eval(keyword.value, line)
        for keyword in node.keywords
    }

    target = kwargs.pop("target", args[0] if args else None)
    name = kwargs.pop("name", args[1] if len(args) > 1 else None)

    if target is None:
        raise TemplateMetadataError(
            f"Directive on line {line} must provide a 'target'"
        )
    if name is None:
        raise TemplateMetadataError(
            f"Directive on line {line} must provide a 'name'"
        )

    if len(args) > 2:
        raise TemplateMetadataError(
            f"Directive on line {line} accepts at most two positional arguments"
        )

    kind = func.attr
    options = _freeze_mapping(kwargs)
    return TemplateDirective(
        kind=kind,
        target=str(target),
        name=str(name),
        options=options,
        source="inline",
        line=line,
    )


def _literal_eval(node: ast.AST, line: int) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
        raise TemplateMetadataError(
            f"Directive on line {line} must use literal arguments"
        ) from exc


def _parse_comment_directives(template: str) -> Iterable[TemplateDirective]:
    for line_number, raw_line in enumerate(template.splitlines(), start=1):
        directive_body: str | None = None
        for marker in ("-- webbed_duck:", "# webbed_duck:"):
            marker_index = raw_line.find(marker)
            if marker_index == -1:
                continue
            directive_body = raw_line[marker_index + len(marker) :].strip()
            break
        if directive_body is None:
            continue
        if not directive_body:
            raise TemplateMetadataError(
                f"Directive at line {line_number} is missing a payload"
            )

        kind, _, remainder = directive_body.partition(" ")
        if not kind:
            raise TemplateMetadataError(
                f"Directive at line {line_number} must include a directive kind"
            )

        options = _parse_options(remainder.strip(), line_number)
        target = _require_option(options, ("target", "path", "field"), line_number)
        name = _require_option(options, ("name", "validator", "rule"), line_number)
        yield TemplateDirective(
            kind=kind,
            target=target,
            name=name,
            options=_freeze_mapping(options),
            source="comment",
            line=line_number,
        )


def _parse_options(body: str, line_number: int) -> MutableMapping[str, Any]:
    if not body:
        return {}
    tokens = shlex.split(body, comments=False, posix=True)
    options: MutableMapping[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise TemplateMetadataError(
                f"Malformed token '{token}' in directive on line {line_number}"
            )
        key, value = token.split("=", 1)
        options[key] = _coerce_value(value)
    return options


def _coerce_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _require_option(options: MutableMapping[str, Any], keys: Sequence[str], line: int) -> str:
    for key in keys:
        if key in options:
            value = options.pop(key)
            return str(value)
    expected = ", ".join(keys)
    raise TemplateMetadataError(
        f"Directive on line {line} must include one of: {expected}"
    )


def _freeze_mapping(mapping: Mapping[str, _T]) -> Mapping[str, _T]:
    return MappingProxyType(dict(mapping))


def _iter_whitelist_directives(
    request_context: Mapping[str, Any] | None,
) -> Iterable[TemplateDirective]:
    parameters = request_context.get("parameters") if request_context else None
    if not isinstance(parameters, Mapping):
        return ()

    directives: list[TemplateDirective] = []
    for group, config in parameters.items():
        if not isinstance(config, Mapping):
            continue
        whitelist_spec = config.get("whitelist")
        if whitelist_spec is None:
            continue
        resolved = _resolve_whitelist(whitelist_spec)
        if resolved is None:
            continue
        label = getattr(whitelist_spec, "label", config.get("label", "whitelist"))
        options: dict[str, Any] = {"values": resolved}
        if label:
            options["label"] = str(label)
        directives.append(
            TemplateDirective(
                kind="whitelist",
                target=f"parameters.{group}",
                name="allowed",
                options=_freeze_mapping(options),
                source="validation",
            )
        )
    return tuple(directives)


def _resolve_whitelist(spec: Any) -> tuple[Any, ...] | None:
    if isinstance(spec, ParameterWhitelist):
        resolved = spec.resolve()
        return tuple(sorted(resolved))
    if isinstance(spec, (set, frozenset)):
        return tuple(sorted(spec))
    if isinstance(spec, (list, tuple)):
        return tuple(spec)
    return None


def _iter_validation_directives(
    validation: Mapping[str, Any] | ValidationContext | None,
) -> Iterable[TemplateDirective]:
    specs = _resolve_parameter_specs(validation)
    if not specs:
        return ()

    directives: list[TemplateDirective] = []
    for name, spec in specs.items():
        directives.extend(_directives_for_spec(spec, parameter=name))
    return tuple(directives)


def _resolve_parameter_specs(
    validation: Mapping[str, Any] | ValidationContext | None,
) -> Mapping[str, ParameterSpec]:
    if validation is None:
        return {}
    if isinstance(validation, ValidationContext):
        return validation.specs
    if not isinstance(validation, Mapping):
        return {}
    parameters = validation.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    specs: dict[str, ParameterSpec] = {}
    for name, data in parameters.items():
        if isinstance(data, Mapping):
            specs[name] = ParameterSpec.from_manifest(name, data)
    return specs


def _directives_for_spec(spec: ParameterSpec, *, parameter: str) -> list[TemplateDirective]:
    if not spec.guards:
        return []

    directives: list[TemplateDirective] = []
    target = f"parameters.{parameter}"
    for guard_name, payload in spec.guards.items():
        if guard_name == "choices":
            values = _as_iterable(payload)
            if values is None:
                continue
            directives.append(
                _validation_directive(target, "choices", {"values": values})
            )
        elif guard_name == "regex" and isinstance(payload, str):
            directives.append(
                _validation_directive(target, "regex", {"pattern": payload})
            )
        elif guard_name == "length" and isinstance(payload, Mapping):
            options = {
                key: payload[key]
                for key in ("min", "max")
                if payload.get(key) is not None
            }
            if options:
                directives.append(
                    _validation_directive(target, "length", options)
                )
        elif guard_name == "range" and isinstance(payload, Mapping):
            options = {
                key: payload[key]
                for key in ("min", "max")
                if payload.get(key) is not None
            }
            if options:
                directives.append(
                    _validation_directive(target, "range", options)
                )
        elif guard_name == "datetime_window" and isinstance(payload, Mapping):
            options = {
                key: payload[key]
                for key in ("earliest", "latest")
                if payload.get(key) is not None
            }
            if options:
                directives.append(
                    _validation_directive(target, "datetime_window", options)
                )
        elif guard_name == "compare" and isinstance(payload, Mapping):
            other = payload.get("parameter")
            operator = payload.get("operator")
            options: dict[str, Any] = {}
            if other is not None:
                options["parameter"] = other
            if operator is not None:
                options["operator"] = operator
            if options:
                directives.append(
                    _validation_directive(target, "compare", options)
                )
    return directives


def _validation_directive(
    target: str, name: str, options: Mapping[str, Any]
) -> TemplateDirective:
    return TemplateDirective(
        kind="validator",
        target=target,
        name=name,
        options=_freeze_mapping(options),
        source="validation",
    )


def _as_iterable(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(value))
    return None


def _looks_like_sql_template(path: Path) -> bool:
    suffixes = {suffix.lstrip(".") for suffix in path.suffixes}
    return "sql" in suffixes


def _slug_for_template(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    while relative.suffix:
        relative = relative.with_suffix("")
    return relative.as_posix()
