"""Tests for the parameter binding helpers."""

from __future__ import annotations

import pytest

from webbed_duck._templating.binding import (
    ParameterBindingError,
    ValidationContext,
)
from webbed_duck._templating.errors import TemplateApplicationError
from webbed_duck._templating.renderer import TemplateRenderer


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

