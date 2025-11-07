"""State helpers for preparing templating contexts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping

from .errors import TemplateApplicationError
from .formatters import (
    DEFAULT_DATE_FORMATS,
    DEFAULT_NUMBER_FORMATS,
    DEFAULT_TIMESTAMP_FORMATS,
    merge_formats,
)
from .parameters import ParameterWhitelist, StringParameterWhitelist

__all__ = [
    "StringNamespace",
    "ParameterWhitelist",
    "StringParameterWhitelist",
    "PreparedContext",
    "RequestContextStore",
    "prepare_context",
]


class StringNamespace(dict):
    """Dictionary enforcing a whitelist for string constants."""

    def __init__(self, data: Mapping[str, Any], whitelist: Iterable[str]):
        super().__init__(data)
        self._whitelist = frozenset(whitelist)
        invalid_keys = set(self) - self._whitelist
        if invalid_keys:
            invalid = next(iter(invalid_keys))
            raise TemplateApplicationError(
                f"String constant '{invalid}' is not present in the whitelist"
            )

    def _ensure_allowed(self, key: str) -> None:
        if key not in self._whitelist:
            raise TemplateApplicationError(
                f"String constant '{key}' is not present in the whitelist"
            )

    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        self._ensure_allowed(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        return super().get(key, default) if key in self._whitelist else default


@dataclass(frozen=True)
class PreparedContext:
    """Normalized view over the request context."""

    constants: Mapping[str, Any]
    parameters: Mapping[str, Any]
    date_formats: Mapping[str, Any]
    timestamp_formats: Mapping[str, Any]
    number_formats: Mapping[str, Any]


class RequestContextStore:
    """Small helper object storing the ambient request context."""

    __slots__ = ("_context",)

    def __init__(self) -> None:
        self._context: Mapping[str, Any] = {
            "constants": {
                "str": {},
                "date": {},
                "timestamp": {},
                "number": {},
                "misc": {},
            },
            "parameters": {
                "str": {"whitelist": frozenset()},
                "date": {"format": {}},
                "timestamp": {"format": {}},
                "number": {"format": {}},
            },
        }

    def get(self) -> Mapping[str, Any]:
        return self._context

    def set(self, context: Mapping[str, Any]) -> None:
        self._context = ensure_mapping(context)


def ensure_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    raise TemplateApplicationError("Request context must be a mapping")


def prepare_context(request_context: Mapping[str, Any]) -> PreparedContext:
    context_mapping = ensure_mapping(request_context)
    constants: MutableMapping[str, Any] = dict(context_mapping.get("constants", {}))
    raw_parameters = context_mapping.get("parameters", {})

    template_parameters: Mapping[str, Any]
    parameters_config: MutableMapping[str, Any]

    if hasattr(raw_parameters, "for_template") and callable(
        getattr(raw_parameters, "for_template")
    ):
        template_parameters = raw_parameters.for_template()
        config_mapping = getattr(raw_parameters, "configuration", {})
        if not isinstance(config_mapping, Mapping):
            parameters_config = {}
        else:
            parameters_config = copy.deepcopy(dict(config_mapping))
    else:
        parameters_config = dict(raw_parameters)
        template_parameters = copy.deepcopy(parameters_config)

    whitelist_spec = parameters_config.get("str", {}).get("whitelist", frozenset())
    whitelist = _resolve_string_whitelist(whitelist_spec)

    str_constants = dict(constants.get("str", {}))
    constants["str"] = StringNamespace(str_constants, whitelist)

    date_formats = merge_formats(
        DEFAULT_DATE_FORMATS, parameters_config.get("date", {}).get("format")
    )
    timestamp_formats = merge_formats(
        DEFAULT_TIMESTAMP_FORMATS, parameters_config.get("timestamp", {}).get("format")
    )
    number_formats = merge_formats(
        DEFAULT_NUMBER_FORMATS, parameters_config.get("number", {}).get("format")
    )

    constants_copy = copy.deepcopy(constants)
    return PreparedContext(
        constants=constants_copy,
        parameters=template_parameters,
        date_formats=date_formats,
        timestamp_formats=timestamp_formats,
        number_formats=number_formats,
    )


def _resolve_string_whitelist(spec: Any) -> frozenset[str]:
    if isinstance(spec, ParameterWhitelist):
        return spec.resolve()
    if spec is None:
        return frozenset()
    if isinstance(spec, str):
        raise TemplateApplicationError("String whitelist must be an iterable of keys")
    if not isinstance(spec, Iterable):
        raise TemplateApplicationError("String whitelist must be iterable")
    return frozenset(spec)
