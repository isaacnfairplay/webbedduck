from __future__ import annotations

from typing import Any, Mapping

from webbed_duck._templating.parameters import ParameterWhitelist


def resolve_whitelist(spec: Any) -> tuple[Any, ...] | None:
    match spec:
        case ParameterWhitelist():
            resolved = spec.resolve()
            return tuple(sorted(resolved))
        case set() | frozenset():
            return tuple(sorted(spec))
        case list() | tuple():
            return tuple(spec)
        case _:
            return None


def build_whitelist_options(config: Any) -> Mapping[str, Any] | None:
    if not isinstance(config, Mapping):
        return None
    whitelist_spec = config.get("whitelist")
    if whitelist_spec is None:
        return None
    resolved = resolve_whitelist(whitelist_spec)
    if resolved is None:
        return None
    options: dict[str, Any] = {"values": resolved}
    label = getattr(whitelist_spec, "label", config.get("label", "whitelist"))
    if label:
        options["label"] = str(label)
    return options
