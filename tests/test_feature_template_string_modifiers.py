"""Tests for extended string modifiers in the template renderer."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples import build_report_request_context
from webbed_duck.constants import TemplateApplicationError, apply_constants


@pytest.fixture()
def request_context():
    return build_report_request_context()


def render(template: str, request_context):
    return apply_constants(template, request_context=request_context)


def test_lower_modifier_applies_case_fold(request_context):
    template = "{{ ctx.constants.str.report_name | lower() }}"
    assert render(template, request_context) == "daily metrics"


def test_upper_modifier_applies_case_fold(request_context):
    template = "{{ ctx.constants.str.report_name | upper() }}"
    assert render(template, request_context) == "DAILY METRICS"


@pytest.mark.parametrize(
    "constant_path, expected",
    [
        ("report_name", "daily_metrics"),
        ("source_path", "srv_data_reports"),
    ],
    ids=["report_name", "source_path"],
)
def test_identifier_modifier_validates_slug(request_context, constant_path, expected):
    template = f"{{{{ ctx.constants.str.{constant_path} | identifier() }}}}"
    assert render(template, request_context) == expected


def test_identifier_modifier_rejects_invalid_tokens(request_context):
    request_context["constants"]["str"]["report_name"] = "2024 report"
    template = "{{ ctx.constants.str.report_name | identifier() }}"
    with pytest.raises(TemplateApplicationError, match=r"must be a valid identifier"):
        render(template, request_context)


def test_literal_modifier_serialises_strings(request_context):
    request_context["constants"]["str"]["report_name"] = "O'Reilly"
    template = "{{ ctx.constants.str.report_name | literal() }}"
    assert render(template, request_context) == "'O''Reilly'"


@pytest.mark.parametrize(
    "constant_path, expected",
    [
        ("discount", "0.125"),
        ("visitors", "12456"),
    ],
)
def test_literal_modifier_handles_numbers(request_context, constant_path, expected):
    template = f"{{{{ ctx.constants.number.{constant_path} | literal() }}}}"
    assert render(template, request_context) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ({"value": 1}, '{"value": 1}'),
        ([1, 2, 3], "[1, 2, 3]"),
    ],
)
def test_json_modifier_serialises_to_json(request_context, value, expected):
    request_context["constants"].setdefault("misc", {})["payload"] = value
    template = "{{ ctx.constants.misc.payload | json() }}"
    assert render(template, request_context) == expected


def test_literal_modifier_for_none(request_context):
    request_context["constants"].setdefault("misc", {})["payload"] = None
    template = "{{ ctx.constants.misc.payload | literal() }}"
    assert render(template, request_context) == "NULL"
