"""Helpers for extracting inline metadata from SQL templates."""

from __future__ import annotations

import ast
import re
import shlex
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, TypeVar

from webbed_duck._templating.binding import ParameterSpec, ValidationContext
from webbed_duck._templating.parameters import ParameterWhitelist

from .cache import CacheConfig
from .cache_support import InvariantFilter
from ._whitelist import build_whitelist_options

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

    validation_map = validations or {}

    routes: dict[str, RouteDescription] = {}
    for path in sorted(template_root.rglob("*")):
        if not path.is_file() or not _looks_like_sql_template(path):
            continue
        template_text = path.read_text(encoding="utf-8")
        slug = _slug_for_template(path, template_root)
        validation = validation_map.get(slug)
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


_PLACEHOLDER = re.compile(r"{{\s*(webbed_duck\..*?)\s*}}")
_COMMENT_DIRECTIVE = re.compile(r"(?:--|#)\s*webbed_duck:(?P<body>.*)")


def _parse_inline_directives(template: str) -> Iterable[TemplateDirective]:
    for match in _PLACEHOLDER.finditer(template):
        expression = match.group(1).strip()
        line_number = template.count("\n", 0, match.start()) + 1
        directive = _parse_inline_expression(expression, line_number)
        if directive is not None:
            yield directive


def _parse_inline_expression(expression: str, line: int) -> TemplateDirective | None:
    node = _parse_expression(expression, line)
    match node:
        case ast.Call(
            func=ast.Attribute(attr=kind, value=ast.Name(id="webbed_duck")),
            args=call_args,
            keywords=call_keywords,
        ):
            pass
        case _:
            return None

    if any(keyword.arg is None for keyword in call_keywords):
        raise TemplateMetadataError(
            f"Directive on line {line} does not support **kwargs"
        )

    args = [_literal_eval(arg, line) for arg in call_args]
    if len(args) > 2:
        raise TemplateMetadataError(
            f"Directive on line {line} accepts at most two positional arguments"
        )
    kwargs: MutableMapping[str, Any] = {
        keyword.arg: _literal_eval(keyword.value, line)
        for keyword in call_keywords
    }
    target = kwargs.pop("target", args[0] if args else None)
    if target is None:
        raise TemplateMetadataError(
            f"Directive on line {line} must provide a 'target'"
        )
    name = kwargs.pop("name", args[1] if len(args) > 1 else None)
    if name is None:
        raise TemplateMetadataError(
            f"Directive on line {line} must provide a 'name'"
        )

    options = _freeze_mapping(kwargs)
    return TemplateDirective(
        kind=kind,
        target=str(target),
        name=str(name),
        options=options,
        source="inline",
        line=line,
    )


def _parse_expression(expression: str, line: int) -> ast.AST:
    try:
        return ast.parse(expression, mode="eval").body
    except SyntaxError as exc:  # pragma: no cover - surfaced in tests
        raise TemplateMetadataError(
            f"Invalid inline directive on line {line}: {expression!r}"
        ) from exc




def _literal_eval(node: ast.AST, line: int) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
        raise TemplateMetadataError(
            f"Directive on line {line} must use literal arguments"
        ) from exc


def _parse_comment_directives(template: str) -> Iterable[TemplateDirective]:
    for line_number, raw_line in enumerate(template.splitlines(), start=1):
        match = _COMMENT_DIRECTIVE.search(raw_line)
        if match is None:
            continue
        directive_body = match.group("body").strip()
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
        try:
            key, value = token.split("=", 1)
        except ValueError as exc:
            raise TemplateMetadataError(
                f"Malformed token '{token}' in directive on line {line_number}"
            ) from exc
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
        options = build_whitelist_options(config)
        if options is None:
            continue
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
    parameter_mapping = _extract_parameter_mapping(validation)
    if parameter_mapping is None:
        return {}
    return {
        name: ParameterSpec.from_manifest(name, data)
        for name, data in parameter_mapping.items()
        if isinstance(data, Mapping)
    }


def _extract_parameter_mapping(
    validation: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(validation, MappingABC):
        return None
    parameters = validation.get("parameters")
    if not isinstance(parameters, Mapping):
        return None
    return parameters


GuardDirectiveBuilder = Callable[[str, Any], tuple[TemplateDirective, ...]]


def _directives_for_spec(
    spec: ParameterSpec, *, parameter: str
) -> tuple[TemplateDirective, ...]:
    if not spec.guards:
        return ()

    directives: list[TemplateDirective] = []
    target = f"parameters.{parameter}"
    for guard_name, payload in spec.guards.items():
        handler = _GUARD_HANDLERS.get(guard_name)
        if handler is None:
            continue
        directives.extend(handler(target, payload))
    return tuple(directives)


def _choices_guard_directives(
    target: str, payload: Any
) -> tuple[TemplateDirective, ...]:
    values = _as_iterable(payload)
    if values is None:
        return ()
    return (_validation_directive(target, "choices", {"values": values}),)


def _regex_guard_directives(
    target: str, payload: Any
) -> tuple[TemplateDirective, ...]:
    if not isinstance(payload, str):
        return ()
    return (_validation_directive(target, "regex", {"pattern": payload}),)


def _build_mapping_guard_directives(
    target: str,
    payload: Any,
    directive_name: str,
    keys: Sequence[str],
) -> tuple[TemplateDirective, ...]:
    options = _filter_guard_options(payload, keys)
    if not options:
        return ()
    return (_validation_directive(target, directive_name, options),)


def _filter_guard_options(
    payload: Any, keys: Sequence[str]
) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    filtered = dict(
        filter(
            lambda item: item[1] is not None,
            ((key, payload.get(key)) for key in keys),
        )
    )
    return filtered or None


_GUARD_HANDLERS: Mapping[str, GuardDirectiveBuilder] = {
    "choices": _choices_guard_directives,
    "regex": _regex_guard_directives,
    "length": partial(
        _build_mapping_guard_directives, directive_name="length", keys=("min", "max")
    ),
    "range": partial(
        _build_mapping_guard_directives, directive_name="range", keys=("min", "max")
    ),
    "datetime_window": partial(
        _build_mapping_guard_directives,
        directive_name="datetime_window",
        keys=("earliest", "latest"),
    ),
    "compare": partial(
        _build_mapping_guard_directives,
        directive_name="compare",
        keys=("parameter", "operator"),
    ),
}


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
