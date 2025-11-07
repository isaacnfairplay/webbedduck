"""Formatting utilities for the templating engine."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from email.utils import format_datetime
from typing import Any, Callable, Dict, Mapping

from .errors import TemplateApplicationError

__all__ = [
    "DEFAULT_DATE_FORMATS",
    "DEFAULT_TIMESTAMP_FORMATS",
    "DEFAULT_NUMBER_FORMATS",
    "merge_formats",
    "date_offset",
    "format_date",
    "format_timestamp",
    "format_number",
    "stringify",
]

DEFAULT_DATE_FORMATS: Mapping[str, str] = {
    "iso": "ISO",
    "iso8601": "ISO",
    "rfc3339": "ISO",
    "yyyy-mm-dd": "%Y-%m-%d",
    "mm-dd-yy": "%m-%d-%y",
    "dd-mm-yyyy": "%d-%m-%Y",
    "month-name": "%B %d, %Y",
    "day-month-name": "%d %B %Y",
    "compact": "%Y%m%d",
}

DEFAULT_TIMESTAMP_FORMATS: Mapping[str, str] = {
    "iso": "ISO",
    "iso8601": "ISO",
    "rfc3339": "ISO",
    "unix": "UNIX",
    "unix_ms": "UNIX_MS",
    "rfc2822": "RFC2822",
}

DEFAULT_NUMBER_FORMATS: Mapping[str, str] = {
    "integer": "d",
    "decimal": ",.2f",
    "percent": ".0%",
    "currency": ",.2f",
    "scientific": ".3e",
}


_TIMESTAMP_HANDLERS: Mapping[str, Callable[[datetime], str]] = {
    "ISO": lambda value: value.isoformat(),
    "UNIX": lambda value: str(int(value.timestamp())),
    "UNIX_MS": lambda value: str(int(value.timestamp() * 1000)),
    "RFC2822": format_datetime,
}


def merge_formats(builtins: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return ``builtins`` merged with user ``overrides``."""

    return {**builtins, **(overrides or {})}


def date_offset(value: Any, *, days: int = 0, weeks: int = 0) -> Any:
    """Apply an offset to *value* treating it as a :class:`~datetime.date` or datetime."""

    delta = timedelta(days=days, weeks=weeks)
    if isinstance(value, datetime):
        return value + delta
    if isinstance(value, date):
        return value + delta
    raise TemplateApplicationError("date_offset requires a date or datetime value")


def _ensure_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def format_date(value: Any, format_key: str, formats: Mapping[str, Any]) -> str:
    """Format a date or datetime using ``format_key`` from ``formats``."""

    if not isinstance(value, (date, datetime)):
        raise TemplateApplicationError("date_format requires a date or datetime")

    format_definition = formats.get(format_key)
    if format_definition is None:
        raise TemplateApplicationError(f"Unknown date format '{format_key}'")

    if format_definition == "ISO":
        return value.isoformat()

    if isinstance(value, datetime) and value.tzinfo is None:
        value = _ensure_datetime(value).astimezone(timezone.utc)

    return value.strftime(str(format_definition))


def format_timestamp(value: Any, format_key: str, formats: Mapping[str, Any]) -> str:
    """Format timestamps using the configured ``formats`` table."""

    if isinstance(value, (int, float, Decimal)):
        value = datetime.fromtimestamp(float(value), tz=timezone.utc)

    if not isinstance(value, datetime):
        raise TemplateApplicationError("timestamp_format requires a datetime")

    format_definition = formats.get(format_key)
    if format_definition is None:
        raise TemplateApplicationError(f"Unknown timestamp format '{format_key}'")

    value = _ensure_datetime(value)

    if isinstance(format_definition, str):
        handler = _TIMESTAMP_HANDLERS.get(format_definition)
        return handler(value) if handler else value.strftime(format_definition)

    raise TemplateApplicationError(
        f"Unsupported timestamp format definition '{format_definition}'"
    )


def format_number(value: Any, format_key: str, formats: Mapping[str, Any]) -> str:
    """Format numeric values according to ``formats``."""

    if not isinstance(value, (int, float, Decimal)):
        raise TemplateApplicationError("number_format requires a numeric value")

    format_definition = formats.get(format_key)
    if format_definition is None:
        raise TemplateApplicationError(f"Unknown number format '{format_key}'")

    if isinstance(format_definition, Mapping):
        format_definition = format_definition.get("spec")
        if format_definition is None:
            raise TemplateApplicationError(
                f"Invalid number format configuration for '{format_key}'"
            )

    if isinstance(format_definition, str):
        return format(value, format_definition)

    raise TemplateApplicationError(
        f"Unsupported number format definition '{format_definition}'"
    )


def stringify(value: Any) -> str:
    """Serialise primitives and temporal values to template-friendly strings."""

    match value:
        case None:
            return "null"
        case bool() as boolean:
            return "true" if boolean else "false"
        case int() | float() | Decimal():
            return format(value, "g")
        case datetime() as dt:
            return _ensure_datetime(dt).isoformat()
        case date() as current_date:
            return current_date.isoformat()

    return str(value)
