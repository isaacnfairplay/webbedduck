import pathlib

import pytest

from webbed_duck._templating.binding import ParameterBindingError, ValidationContext


@pytest.fixture
def validation_context() -> ValidationContext:
    manifest = {
        "parameters": {
            "asset_path": {
                "type": "string",
                "required": True,
                "allow_template": True,
                "guards": {"path": {}},
            }
        }
    }
    return ValidationContext.from_manifest(manifest)


def test_safe_path_guard_allows_relative_child(validation_context: ValidationContext) -> None:
    provided = {"asset_path": "reports/summary.csv"}
    context = validation_context.resolve(provided)
    assert context["asset_path"].value == "reports/summary.csv"


def test_safe_path_guard_blocks_parent_segments(validation_context: ValidationContext) -> None:
    with pytest.raises(ParameterBindingError, match=r"asset_path.*parent segments"):
        validation_context.resolve({"asset_path": "../secrets.txt"})


def test_safe_path_guard_blocks_absolute_paths(validation_context: ValidationContext) -> None:
    with pytest.raises(ParameterBindingError, match=r"asset_path.*absolute path"):
        validation_context.resolve({"asset_path": "/etc/passwd"})


def test_safe_path_guard_blocks_backslashes(validation_context: ValidationContext) -> None:
    with pytest.raises(ParameterBindingError, match=r"asset_path.*backslashes"):
        validation_context.resolve({"asset_path": "..\\windows\\system32"})


def test_safe_path_guard_can_allow_parent_segments() -> None:
    manifest = {
        "parameters": {
            "asset_path": {
                "type": "string",
                "guards": {"path": {"allow_parent": True}},
            }
        }
    }
    context = ValidationContext.from_manifest(manifest)
    resolved = context.resolve({"asset_path": "../ok.sql"})
    assert resolved["asset_path"].value == "../ok.sql"


def test_safe_path_guard_respects_optional_parameter(validation_context: ValidationContext) -> None:
    manifest = {
        "parameters": {
            "asset_path": {
                "type": "string",
                "required": False,
                "guards": {"path": {}},
            }
        }
    }
    context = ValidationContext.from_manifest(manifest)
    resolved = context.resolve({})
    assert "asset_path" not in resolved


@pytest.mark.parametrize(
    "invalid_config",
    [
        1,
        "not-a-mapping",
        {"allow_parent": "sometimes"},
    ],
)
def test_safe_path_guard_configuration_validation(invalid_config) -> None:
    manifest = {
        "parameters": {
            "asset_path": {
                "type": "string",
                "guards": {"path": invalid_config},
            }
        }
    }
    context = ValidationContext.from_manifest(manifest)
    expected_pattern = r"path guard must be a mapping"
    if isinstance(invalid_config, dict):
        expected_pattern = r"path guard option"
    with pytest.raises(ParameterBindingError, match=expected_pattern):
        context.resolve({"asset_path": "reports/output.csv"})
