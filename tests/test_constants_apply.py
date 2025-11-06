"""Tests for applying request constants using the templating helper."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from webbed_duck.constants import TemplateApplicationError, apply_constants
from webbed_duck.examples import (
    BUILTIN_DATE_FORMAT_EXPECTATIONS,
    BUILTIN_NUMBER_FORMAT_EXPECTATIONS,
    BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS,
    EXAMPLE_TEMPLATE_CASES,
    SQL_TEMPLATE_CASE,
    build_report_request_context,
)


@pytest.fixture
def request_context():
    return build_report_request_context()


@pytest.mark.parametrize("case", EXAMPLE_TEMPLATE_CASES, ids=lambda case: case.slug)
def test_apply_constants_across_supported_types(case, request_context):
    assert apply_constants(case.template, request_context=request_context) == case.expected


def test_date_offset_and_format(request_context):
    template = "{{ ctx.constants.date.run | date_offset(days=3) | date_format('mm-dd-yy') }}"
    assert apply_constants(template, request_context=request_context) == "02-03-24"


def test_extra_context_entries_are_ignored(request_context):
    template = "SELECT '{{ ctx.constants.str.report_name }}'"
    assert apply_constants(template, request_context=request_context) == "SELECT 'Daily Metrics'"


def test_unapproved_string_is_blocked(request_context):
    template = "{{ ctx.constants.str.unapproved }}"
    with pytest.raises(TemplateApplicationError):
        apply_constants(template, request_context=request_context)


def test_invalid_modifier_reports_error(request_context):
    template = "{{ ctx.constants.str.report_name | unknown_modifier() }}"
    with pytest.raises(TemplateApplicationError):
        apply_constants(template, request_context=request_context)


def test_invalid_path_reports_error(request_context):
    template = "{{ ctx.unexpected.value }}"
    with pytest.raises(TemplateApplicationError):
        apply_constants(template, request_context=request_context)


def test_sql_template_round_trip(request_context):
    rendered = apply_constants(SQL_TEMPLATE_CASE.template, request_context=request_context)
    assert rendered == SQL_TEMPLATE_CASE.expected


@pytest.mark.parametrize("template", [
    "{{ ctx.constants.number.discount | number_format('unknown') }}",
    "{{ ctx.constants.date.run | date_format('unknown') }}",
    "{{ ctx.constants.timestamp.created | timestamp_format('unknown') }}",
])
def test_unknown_format_keys_raise(template: str, request_context):
    with pytest.raises(TemplateApplicationError):
        apply_constants(template, request_context=request_context)


@pytest.mark.parametrize(
    "format_key, expected",
    BUILTIN_DATE_FORMAT_EXPECTATIONS.items(),
)
def test_all_known_date_formats(format_key: str, expected: str, request_context):
    template = f"{{{{ ctx.constants.date.run | date_format('{format_key}') }}}}"
    assert apply_constants(template, request_context=request_context) == expected


@pytest.mark.parametrize(
    "format_key, expected",
    BUILTIN_TIMESTAMP_FORMAT_EXPECTATIONS.items(),
)
def test_all_known_timestamp_formats(format_key: str, expected: str, request_context):
    template = f"{{{{ ctx.constants.timestamp.created | timestamp_format('{format_key}') }}}}"
    assert apply_constants(template, request_context=request_context) == expected


@pytest.mark.parametrize(
    "format_key, expected",
    BUILTIN_NUMBER_FORMAT_EXPECTATIONS.items(),
)
def test_common_number_formats(format_key: str, expected: str, request_context):
    constant_path = "discount" if format_key == "percent" else "visitors"
    template = f"{{{{ ctx.constants.number.{constant_path} | number_format('{format_key}') }}}}"
    assert apply_constants(template, request_context=request_context) == expected
