"""Tests for the parameter binding helpers."""

from __future__ import annotations

import pytest

from examples import build_report_request_context
from webbed_duck._templating.binding import (
    ParameterBindingError,
    ValidationContext,
)
from webbed_duck._templating.errors import TemplateApplicationError
from webbed_duck._templating.renderer import TemplateRenderer
from webbed_duck._templating.state import prepare_context


def build_validation(manifest: dict[str, object]) -> ValidationContext:
    return ValidationContext.from_manifest(manifest)


def test_parameter_binding_coercion_and_defaults(tmp_path) -> None:
    validation = build_validation(
        {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "default": 25,
                    "allow_template": True,
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                },
                "search": {
                    "type": "string",
                    "default": "duck",
                    "allow_template": True,
                },
            }
        }
    )

    context = validation.resolve({"limit": "10", "verbose": "true"})

    assert context["limit"].value == 10
    assert context["limit"].provenance == "provided"

    assert context["verbose"].value is True
    assert context["verbose"].provenance == "provided"

    assert context["search"].value == "duck"
    assert context["search"].provenance == "default"

    template_view = context.for_template()
    assert template_view["limit"] == 10
    assert template_view.get("search") == "duck"

    assert context.template_consumed == {"limit", "search"}

    binding = context.for_binding()
    assert binding == {"verbose": True}

    projected_binding = context.for_binding(used_names={"verbose"})
    assert projected_binding == {"verbose": True}


def test_parameter_binding_tracks_template_usage_via_renderer() -> None:
    validation = build_validation(
        {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "allow_template": True,
                },
                "offset": {
                    "type": "integer",
                },
            }
        }
    )

    context = validation.resolve({"limit": 5, "offset": 2})
    renderer = TemplateRenderer({"parameters": context})

    rendered = renderer.render("SELECT {{ ctx.parameters.limit }}")

    assert rendered == "SELECT 5"
    assert context.template_consumed == {"limit"}

    binding = context.for_binding()
    assert binding == {"offset": 2}


def test_missing_required_parameter() -> None:
    validation = build_validation(
        {
            "parameters": {
                "limit": {"type": "integer", "required": True}
            }
        }
    )

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({})

    assert "Parameter 'limit' is required" in str(excinfo.value)


def test_guard_failures() -> None:
    validation = build_validation(
        {
            "parameters": {
                "category": {
                    "type": "string",
                    "guards": {"choices": ["duck", "goose"]},
                },
                "code": {
                    "type": "string",
                    "guards": {"regex": r"^[A-Z]{2}$"},
                },
                "tag": {
                    "type": "string",
                    "guards": {"length": {"min": 2, "max": 4}},
                },
            }
        }
    )

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({
            "category": "swan",
            "code": "ABC",
            "tag": "a",
        })

    message = str(excinfo.value)
    assert "Parameter 'category' must be one of" in message
    assert "Parameter 'code' must match pattern" in message
    assert "Parameter 'tag' length" in message


def test_numeric_range_guard() -> None:
    validation = build_validation(
        {
            "parameters": {
                "score": {
                    "type": "number",
                    "guards": {"range": {"min": 0, "max": 1}},
                }
            }
        }
    )

    context = validation.resolve({"score": 0.5})
    assert context["score"].value == pytest.approx(0.5)

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({"score": 2})

    assert "Parameter 'score' must be between 0 and 1" in str(excinfo.value)


def test_datetime_window_guard() -> None:
    validation = build_validation(
        {
            "parameters": {
                "published": {
                    "type": "string",
                    "guards": {
                        "datetime_window": {
                            "earliest": "2023-01-01T00:00:00",
                            "latest": "2023-12-31T23:59:59",
                        }
                    },
                }
            }
        }
    )

    context = validation.resolve({"published": "2023-06-01T12:00:00"})
    assert context["published"].value == "2023-06-01T12:00:00"

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({"published": "2024-01-01T00:00:00"})

    assert "Parameter 'published' must not be later than" in str(excinfo.value)

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({"published": "2022-12-31T23:59:59"})

    assert "Parameter 'published' must not be earlier than" in str(excinfo.value)


def test_datetime_window_guard_timezone_mismatch() -> None:
    validation = build_validation(
        {
            "parameters": {
                "published": {
                    "type": "string",
                    "guards": {
                        "datetime_window": {
                            "earliest": "2023-01-01T00:00:00+00:00",
                        }
                    },
                }
            }
        }
    )

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({"published": "2023-06-01T00:00:00"})

    assert "timezone awareness" in str(excinfo.value)


def test_cross_field_compare_guard() -> None:
    validation = build_validation(
        {
            "parameters": {
                "start": {"type": "integer", "required": True},
                "end": {
                    "type": "integer",
                    "guards": {
                        "compare": {
                            "parameter": "start",
                            "operator": "gte",
                        }
                    },
                },
            }
        }
    )

    context = validation.resolve({"start": 5, "end": 7})
    assert context["end"].value == 7

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({"start": 5, "end": 3})

    assert (
        "Parameter 'end' must be greater than or equal to parameter 'start'"
        in str(excinfo.value)
    )
def test_template_whitelist_blocks_unapproved_parameters() -> None:
    validation = build_validation(
        {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "allow_template": True,
                },
                "offset": {
                    "type": "integer",
                    "allow_template": False,
                },
            }
        }
    )

    context = validation.resolve({"limit": 1, "offset": 2})
    template_view = context.for_template()

    assert template_view["limit"] == 1

    with pytest.raises(TemplateApplicationError):
        _ = template_view["offset"]


def test_binding_projection_filters_unused_parameters() -> None:
    validation = build_validation(
        {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "allow_template": True,
                },
                "offset": {
                    "type": "integer",
                },
                "category": {
                    "type": "string",
                },
            }
        }
    )

    context = validation.resolve({
        "limit": 5,
        "offset": 2,
        "category": "duck",
    })

    context.for_template()["limit"]
    binding = context.for_binding(used_names={"offset"})

    assert binding == {"offset": 2}


def test_unknown_parameters_rejected_by_default() -> None:
    validation = build_validation(
        {
            "parameters": {
                "known": {
                    "type": "integer",
                }
            }
        }
    )

    with pytest.raises(ParameterBindingError) as excinfo:
        validation.resolve({"dynamic": "value"})

    assert "Unknown parameter 'dynamic'" in str(excinfo.value)


def test_binding_respects_unknown_parameter_policy() -> None:
    validation = build_validation(
        {
            "parameters": {
                "known": {
                    "type": "integer",
                    "allow_template": True,
                }
            },
            "allow_unknown_parameters": True,
        }
    )

    context = validation.resolve({"known": 1, "dynamic": "value"})

    template_view = context.for_template()
    assert template_view["known"] == 1

    with pytest.raises(TemplateApplicationError):
        _ = template_view["dynamic"]

    binding = context.for_binding()
    assert binding == {"dynamic": "value"}

    filtered = context.for_binding(used_names={"dynamic"})
    assert filtered == {"dynamic": "value"}


def test_parameter_context_preserves_whitelist_for_constants() -> None:
    request_context = build_report_request_context()
    validation = build_validation(
        {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "allow_template": True,
                }
            }
        }
    )

    parameter_context = validation.resolve({"limit": 5}).with_configuration(
        request_context["parameters"]
    )

    prepared = prepare_context(
        {
            "constants": request_context["constants"],
            "parameters": parameter_context,
        }
    )

    assert prepared.constants["str"]["report_name"] == "Daily Metrics"
    with pytest.raises(TemplateApplicationError):
        _ = prepared.constants["str"]["unapproved"]

