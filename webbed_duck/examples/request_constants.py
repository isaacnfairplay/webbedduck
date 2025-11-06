"""Example request context and template cases for documentation and tests."""

from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Mapping, Sequence

__all__ = [
    "TemplateCase",
    "build_report_request_context",
    "BUILTIN_DATE_FORMAT_EXPECTATIONS",
    "BUILTIN_NUMBER_FORMAT_EXPECTATIONS",
    "BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS",
    "BUILTIN_TEMPLATE_CASES",
    "EXAMPLE_TEMPLATE_CASES",
    "SQL_TEMPLATE_CASE",
]


@dataclass(frozen=True)
class TemplateCase:
    """Simple structure describing a template and its expected output."""

    slug: str
    template: str
    expected: str
    description: str = ""


_BASE_DATE = dt.date(2024, 1, 31)
_BASE_TIMESTAMP = dt.datetime(2024, 1, 31, 12, 15, 33, tzinfo=dt.timezone.utc)

_BASE_CONTEXT: Dict[str, Any] = {
    "constants": {
        "str": {
            "source_path": "/srv/data/reports",
            "report_name": "Daily Metrics",
        },
        "date": {
            "run": _BASE_DATE,
        },
        "timestamp": {
            "created": _BASE_TIMESTAMP,
        },
        "number": {
            "discount": Decimal("0.125"),
            "visitors": 12456,
        },
        "misc": {
            "optional": None,
            "active": True,
        },
    },
    "parameters": {
        "str": {"whitelist": {"source_path", "report_name"}},
        "date": {
            "format": {
                "yyyy-mm-dd": "%Y-%m-%d",
                "mm-dd-yy": "%m-%d-%y",
                "month-name": "%B %d, %Y",
            }
        },
        "timestamp": {
            "format": {
                "iso": "ISO",
                "unix": "UNIX",
                "unix_ms": "UNIX_MS",
            }
        },
        "number": {
            "format": {
                "percent": {"spec": ".0%"},
                "decimal": ",.3f",
            }
        },
    },
    "extra": {"ignored": "value"},
}


BUILTIN_DATE_FORMAT_EXPECTATIONS = {
    "iso": "2024-01-31",
    "iso8601": "2024-01-31",
    "rfc3339": "2024-01-31",
    "yyyy-mm-dd": "2024-01-31",
    "mm-dd-yy": "01-31-24",
    "dd-mm-yyyy": "31-01-2024",
    "month-name": "January 31, 2024",
    "day-month-name": "31 January 2024",
    "compact": "20240131",
}

BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS = {
    "iso": "2024-01-31T12:15:33+00:00",
    "iso8601": "2024-01-31T12:15:33+00:00",
    "rfc3339": "2024-01-31T12:15:33+00:00",
    "unix": str(int(_BASE_TIMESTAMP.timestamp())),
    "unix_ms": str(int(_BASE_TIMESTAMP.timestamp() * 1000)),
    "rfc2822": "Wed, 31 Jan 2024 12:15:33 +0000",
}

BUILTIN_NUMBER_FORMAT_EXPECTATIONS = {
    "integer": "12456",
    "decimal": "12,456.000",
    "percent": "12%",
    "currency": "12,456.00",
    "scientific": "1.246e+04",
}


def build_report_request_context() -> Mapping[str, Any]:
    """Return a deep copy of the example request context."""

    return copy.deepcopy(_BASE_CONTEXT)


EXAMPLE_TEMPLATE_CASES: Sequence[TemplateCase] = (
    TemplateCase(
        slug="string_source_path",
        template="{{ ctx.constants.str.source_path }}",
        expected=_BASE_CONTEXT["constants"]["str"]["source_path"],
        description="Fetches a whitelisted string constant.",
    ),
    TemplateCase(
        slug="string_report_name",
        template="{{ ctx.constants.str.report_name }}",
        expected=_BASE_CONTEXT["constants"]["str"]["report_name"],
        description="Fetches another whitelisted string constant.",
    ),
    TemplateCase(
        slug="date_format_yyyy_mm_dd",
        template="{{ ctx.constants.date.run | date_format('yyyy-mm-dd') }}",
        expected="2024-01-31",
        description="Formats the run date using a custom context override.",
    ),
    TemplateCase(
        slug="timestamp_unix",
        template="{{ ctx.constants.timestamp.created | timestamp_format('unix') }}",
        expected=BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS["unix"],
        description="Renders the created timestamp as a UNIX second string.",
    ),
    TemplateCase(
        slug="timestamp_unix_ms",
        template="{{ ctx.constants.timestamp.created | timestamp_format('unix_ms') }}",
        expected=BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS["unix_ms"],
        description="Renders the created timestamp as a UNIX millisecond string.",
    ),
    TemplateCase(
        slug="number_percent",
        template="{{ ctx.constants.number.discount | number_format('percent') }}",
        expected=BUILTIN_NUMBER_FORMAT_EXPECTATIONS["percent"],
        description="Shows a discount value as a rounded percentage.",
    ),
    TemplateCase(
        slug="number_decimal",
        template="{{ ctx.constants.number.visitors | number_format('decimal') }}",
        expected=BUILTIN_NUMBER_FORMAT_EXPECTATIONS["decimal"],
        description="Applies a decimal formatter override for visitors.",
    ),
    TemplateCase(
        slug="misc_optional_null",
        template="{{ ctx.constants.misc.optional }}",
        expected="null",
        description="Demonstrates null serialisation in the renderer.",
    ),
    TemplateCase(
        slug="misc_active_bool",
        template="{{ ctx.constants.misc.active }}",
        expected="true",
        description="Demonstrates boolean serialisation in the renderer.",
    ),
)

SQL_TEMPLATE_CASE = TemplateCase(
    slug="sql_round_trip",
    template=(
        "SELECT * FROM '{{ ctx.constants.str.source_path }}/metrics.csv' "
        "WHERE run_date = '{{ ctx.constants.date.run | date_format('month-name') }}' "
        "AND created_at >= '{{ ctx.constants.timestamp.created | timestamp_format('iso') }}'"
    ),
    expected=(
        "SELECT * FROM '/srv/data/reports/metrics.csv' WHERE run_date = 'January 31, 2024' "
        "AND created_at >= '2024-01-31T12:15:33+00:00'"
    ),
    description="End-to-end SQL snippet using multiple template segments.",
)


def _format_slug(prefix: str, key: str) -> str:
    return f"{prefix}_{key.replace('-', '_')}"


_EXAMPLE_SLUGS = {case.slug for case in EXAMPLE_TEMPLATE_CASES}


def _build_builtin_cases() -> Sequence[TemplateCase]:
    cases: list[TemplateCase] = []
    for key, expected in BUILTIN_DATE_FORMAT_EXPECTATIONS.items():
        slug = _format_slug("date_format", key)
        if slug in _EXAMPLE_SLUGS:
            continue
        cases.append(
            TemplateCase(
                slug=slug,
                template=f"{{{{ ctx.constants.date.run | date_format('{key}') }}}}",
                expected=expected,
                description=f"Applies the '{key}' date formatter.",
            )
        )

    for key, expected in BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS.items():
        slug = _format_slug("timestamp_format", key)
        if slug in _EXAMPLE_SLUGS:
            continue
        cases.append(
            TemplateCase(
                slug=slug,
                template=f"{{{{ ctx.constants.timestamp.created | timestamp_format('{key}') }}}}",
                expected=expected,
                description=f"Applies the '{key}' timestamp formatter.",
            )
        )

    for key, expected in BUILTIN_NUMBER_FORMAT_EXPECTATIONS.items():
        slug = _format_slug("number_format", key)
        if slug in _EXAMPLE_SLUGS:
            continue
        value_path = "discount" if key == "percent" else "visitors"
        cases.append(
            TemplateCase(
                slug=slug,
                template=(
                    f"{{{{ ctx.constants.number.{value_path} | number_format('{key}') }}}}"
                ),
                expected=expected,
                description=f"Applies the '{key}' number formatter.",
            )
        )

    return tuple(cases)


BUILTIN_TEMPLATE_CASES: Sequence[TemplateCase] = _build_builtin_cases()
