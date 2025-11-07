"""Tests for the template filter pipeline feature."""

from __future__ import annotations

import copy

import pytest

from examples import build_report_request_context
from webbed_duck.constants import TemplateApplicationError, apply_constants


@pytest.fixture
def request_context():
    return build_report_request_context()


def test_upper_filter_converts_text_to_uppercase(request_context):
    template = "{{ ctx.constants.str.report_name | upper }}"
    assert apply_constants(template, request_context=request_context) == "DAILY METRICS"


def test_lower_filter_rejects_non_text(request_context):
    template = "{{ ctx.constants.misc.active | lower }}"
    with pytest.raises(TemplateApplicationError, match="requires a text value"):
        apply_constants(template, request_context=request_context)


def test_identifier_filter_rejects_invalid_values(request_context):
    template = "{{ ctx.constants.str.report_name | identifier }}"
    with pytest.raises(TemplateApplicationError, match="must be a valid identifier"):
        apply_constants(template, request_context=request_context)


def test_identifier_filter_allows_valid_identifiers(request_context):
    context = copy.deepcopy(request_context)
    context["constants"]["str"]["report_name"] = "daily_metrics"
    template = "{{ ctx.constants.str.report_name | identifier }}"
    assert apply_constants(template, request_context=context) == "daily_metrics"


def test_literal_filter_quotes_strings_and_lists(request_context):
    context = copy.deepcopy(request_context)
    context["constants"]["misc"]["choices"] = ["alpha", "beta"]
    template = (
        "{{ ctx.constants.str.report_name | literal }}"
        " {{ ctx.constants.misc.choices | literal }}"
    )
    rendered = apply_constants(template, request_context=context)
    assert rendered == "'Daily Metrics' ('alpha', 'beta')"


def test_literal_filter_handles_nulls_and_booleans(request_context):
    template = (
        "{{ ctx.constants.misc.optional | literal }} "
        "{{ ctx.constants.misc.active | literal }}"
    )
    rendered = apply_constants(template, request_context=request_context)
    assert rendered == "NULL TRUE"


def test_json_filter_serialises_values(request_context):
    template = "{{ ctx.constants.str.report_name | json }}"
    assert apply_constants(template, request_context=request_context) == '"Daily Metrics"'
