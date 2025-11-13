from __future__ import annotations

import json
from pathlib import Path
import textwrap
from typing import Any, Iterable, Mapping

import pytest

from examples import build_report_request_context

from webbed_duck._templating.parameters import ParameterWhitelist
from webbed_duck.server.cache import CacheConfig
from webbed_duck.server.cache_support import InvariantFilter
from webbed_duck.server.template_metadata import (
    RouteDescription,
    TemplateMetadataError,
    build_route_registry,
    collect_template_metadata,
)


def test_collect_template_metadata_captures_comments_validation_and_whitelist() -> None:
    template = """
    SELECT * FROM table
    LIMIT {{ ctx.parameters.number.limit }}{{ webbed_duck.validator("parameters.limit", "positive", severity="error") }}
    """

    request_context = {
        "parameters": {
            "str": {
                "whitelist": ParameterWhitelist(
                    requested=["region", "country"],
                    allowed=["region", "country", "channel"],
                    label="String whitelist",
                )
            }
        }
    }
    validation_manifest = {
        "parameters": {
            "country": {
                "type": "string",
                "guards": {"choices": ["US", "CA"]},
            },
            "limit": {
                "type": "integer",
                "guards": {"range": {"min": 1, "max": 500}},
            },
        }
    }

    metadata = collect_template_metadata(
        template,
        request_context=request_context,
        validation=validation_manifest,
    )

    assert len(metadata.directives) == 4
    inline_directive = metadata.directives[0]
    assert inline_directive.kind == "validator"
    assert inline_directive.source == "inline"
    assert inline_directive.options["severity"] == "error"

    whitelist_directive = next(
        directive for directive in metadata.directives if directive.kind == "whitelist"
    )
    assert whitelist_directive.target == "parameters.str"
    assert whitelist_directive.options["label"] == "String whitelist"
    assert whitelist_directive.options["values"] == ("country", "region")

    validator_targets = {
        (directive.target, directive.name): directive.options
        for directive in metadata.directives
        if directive.source == "validation"
    }
    assert ("parameters.country", "choices") in validator_targets
    assert validator_targets[("parameters.country", "choices")]["values"] == (
        "US",
        "CA",
    )
    assert ("parameters.limit", "range") in validator_targets
    assert validator_targets[("parameters.limit", "range")]["min"] == 1


def test_collect_template_metadata_requires_targets() -> None:
    template = "{{ webbed_duck.validator(name='positive') }}"

    with pytest.raises(TemplateMetadataError):
        collect_template_metadata(template)


def test_build_route_registry_combines_metadata_and_invariants(tmp_path: Path) -> None:
    template_root = tmp_path / "routes"
    template_root.mkdir()
    template_path = template_root / "reports" / "sales.sql"
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        """
        SELECT 1
        LIMIT {{ ctx.parameters.number.limit }}{{ webbed_duck.validator("parameters.limit", "positive") }}
        """,
        encoding="utf-8",
    )

    cache_config = CacheConfig(
        invariants={
            "region": InvariantFilter(key="region", case_insensitive=True),
        }
    )

    request_context = {
        "parameters": {
            "str": {
                "whitelist": ParameterWhitelist(
                    requested=["region"],
                    allowed=["region", "channel"],
                )
            }
        }
    }
    validations = {
        "reports/sales": {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "guards": {"range": {"min": 1, "max": 1000}},
                }
            }
        }
    }

    registry = build_route_registry(
        template_root,
        cache_config=cache_config,
        request_context=request_context,
        validations=validations,
    )

    assert set(registry.keys()) == {"reports/sales"}
    description = registry["reports/sales"]
    assert description.slug == "reports/sales"
    assert description.template_path == template_path
    assert "region" in description.invariants
    assert description.invariants["region"].case_insensitive is True
    assert description.validation is not None
    assert set(description.validation.specs) == {"limit"}

    validators = description.metadata.validators
    validator_names = {(directive.target, directive.name) for directive in validators}
    assert ("parameters.limit", "positive") in validator_names
    assert ("parameters.limit", "range") in validator_names


def test_route_registry_examples_documented() -> None:
    project_root = Path(__file__).resolve().parents[2]
    template_root = project_root / "examples" / "route_templates"
    cache_config = CacheConfig(
        invariants={
            "region": InvariantFilter(
                key="region", column="region_code", case_insensitive=True
            )
        }
    )

    request_context = build_report_request_context()
    parameters = dict(request_context.get("parameters", {}))
    parameters.setdefault("str", {})
    parameters["str"]["whitelist"] = ParameterWhitelist(
        requested=["CA", "US"],
        allowed=["CA", "GB", "US"],
        label="Allowed countries",
    )
    request_context = dict(request_context)
    request_context["parameters"] = parameters

    validations = {
        "catalog/sales_overview": {
            "parameters": {
                "country": {
                    "type": "string",
                    "guards": {"choices": ["CA", "GB", "US"]},
                },
                "limit": {
                    "type": "integer",
                    "guards": {"range": {"min": 1, "max": 500}},
                },
            }
        }
    }

    registry = build_route_registry(
        template_root,
        cache_config=cache_config,
        request_context=request_context,
        validations=validations,
    )

    catalog_only = {
        slug: registry[slug]
        for slug in registry
        if slug.startswith("catalog/")
    }

    document = _render_registry_documentation(catalog_only, template_root=template_root)
    expected_path = project_root / "test_docs" / "cache_route_metadata_examples.md"
    expected = expected_path.read_text(encoding="utf-8")
    assert document == expected


def test_template_metadata_dsl_reference_documented() -> None:
    project_root = Path(__file__).resolve().parents[2]
    template_root = project_root / "examples" / "route_templates"
    template_path = template_root / "reference" / "inline_metadata.sql"

    cache_config = CacheConfig(
        invariants={
            "region": InvariantFilter(
                key="region",
                column="region_code",
                separator="|",
                case_insensitive=True,
            )
        }
    )

    request_context = {
        "parameters": {
            "str": {
                "whitelist": ParameterWhitelist(
                    requested=["CA", "US"],
                    allowed=["CA", "GB", "US"],
                    label="Allowed countries",
                )
            }
        }
    }

    validation_manifest = {
        "parameters": {
            "country": {
                "type": "string",
                "guards": {"choices": ["CA", "GB", "US"]},
            },
            "limit": {
                "type": "integer",
                "guards": {"range": {"min": 1, "max": 1000}},
            },
        }
    }

    document = _render_dsl_reference_documentation(
        template_path=template_path,
        template_root=template_root,
        cache_config=cache_config,
        request_context=request_context,
        validation_manifest=validation_manifest,
    )

    expected_path = project_root / "test_docs" / "template_metadata_dsl_reference.md"
    expected = expected_path.read_text(encoding="utf-8")
    assert document == expected


def _render_registry_documentation(
    registry: Mapping[str, RouteDescription], *, template_root: Path
) -> str:
    lines = ["# Cache route metadata examples", ""]
    for slug in sorted(registry):
        description = registry[slug]
        relative_path = description.template_path.relative_to(template_root)
        lines.append(f"## {slug}")
        lines.append("")
        lines.append(f"* Template: `{relative_path.as_posix()}`")
        lines.append("")
        lines.append("### Invariants")
        invariants = description.invariants
        if invariants:
            for name, invariant in sorted(invariants.items()):
                lines.append(
                    "- "
                    f"{name} → column=`{invariant.column}`, "
                    f"separator=`{invariant.separator}`, "
                    f"case_insensitive={str(invariant.case_insensitive).lower()}"
                )
        else:
            lines.append("- (none)")
        lines.append("")
        lines.append("### Directives")
        directives = sorted(
            description.metadata.directives,
            key=lambda directive: (
                directive.kind,
                directive.target,
                directive.name,
                directive.source,
                directive.line or 0,
            ),
        )
        if directives:
            for directive in directives:
                options = json.dumps(
                    {
                        key: _normalise_option_value(value)
                        for key, value in sorted(directive.options.items())
                    },
                    sort_keys=True,
                )
                lines.append(
                    "- "
                    f"{directive.source} {directive.kind} {directive.target} "
                    f"{directive.name} {options}"
                )
        else:
            lines.append("- (none)")
        lines.append("")
    return "\n".join(lines) + "\n"


def _render_dsl_reference_documentation(
    *,
    template_path: Path,
    template_root: Path,
    cache_config: CacheConfig,
    request_context: Mapping[str, Any],
    validation_manifest: Mapping[str, Any],
) -> str:
    template_text = template_path.read_text(encoding="utf-8")
    metadata = collect_template_metadata(
        template_text,
        request_context=request_context,
        validation=validation_manifest,
    )

    slug = _slug_for_template_path(template_path, template_root)

    registry = build_route_registry(
        template_root,
        cache_config=cache_config,
        request_context=request_context,
        validations={slug: validation_manifest},
    )
    description = registry[slug]
    relative_path = template_path.relative_to(template_root)

    lines = [
        "# Template metadata inline DSL",
        "",
        "The `webbed_duck.server.template_metadata` module extracts structured route metadata",
        "from SQL templates without affecting the rendered SQL. Inline directives live next",
        "to the expressions they protect so reviewers can see why a guard exists while",
        "reading the template.",
        "",
        "## Call anatomy",
        "",
        "Inline directives use call syntax inside template placeholders:",
        "",
        "```jinja",
        "{{ webbed_duck.validator(target, name, **options) }}",
        "```",
        "",
        "- `target` identifies the object being described, such as `parameters.limit`.",
        "- `name` labels the validator rule that applies, for example `positive` or `range`.",
        "- Additional keyword arguments become metadata options. Arguments must be literals",
        "  so metadata can be collected without evaluating arbitrary code.",
        "",
        "The call renders as an empty string, keeping the SQL passed to the database unchanged",
        "while still advertising the guard to tooling and documentation.",
        "",
        f"## Example: `{slug}`",
        "",
        f"* Template: `{relative_path.as_posix()}`",
        "",
        "```sql",
        textwrap.dedent(template_text).strip(),
        "```",
        "",
        "### Inline directives captured from the template",
    ]

    inline_directives = [
        directive for directive in metadata.directives if directive.source == "inline"
    ]
    lines.extend(_format_directive_lines(inline_directives))

    lines.append("")
    lines.append("### Derived directives from validation manifests and whitelists")
    validation_directives = [
        directive for directive in metadata.directives if directive.source == "validation"
    ]
    lines.extend(_format_directive_lines(validation_directives))

    lines.append("")
    lines.append("### Route registry aggregation")
    lines.append(
        "`build_route_registry` merges inline annotations with cache invariants so API"
    )
    lines.append(
        "consumers receive a complete description of each route. For this template the"
    )
    lines.append("registry includes:")
    lines.append("")
    lines.append("**Invariants**")
    if description.invariants:
        for name, invariant in sorted(description.invariants.items()):
            lines.append(
                "- "
                f"{name} → column=`{invariant.column}`, separator=`{invariant.separator}`, "
                f"case_insensitive={str(invariant.case_insensitive).lower()}"
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("**All metadata directives**")
    lines.extend(_format_directive_lines(description.metadata.directives))
    lines.append("")
    lines.append(
        "This document is regenerated by the test suite, ensuring the examples stay in"
    )
    lines.append("sync with the actual parsing and registry behaviour.")
    lines.append("")

    return "\n".join(lines)


def _format_directive_lines(directives: Iterable[TemplateDirective]) -> list[str]:
    lines: list[str] = []
    for directive in _sorted_directives(directives):
        options = json.dumps(
            {
                key: _normalise_option_value(value)
                for key, value in sorted(directive.options.items())
            },
            sort_keys=True,
        )
        lines.append(
            "- "
            f"{directive.source} {directive.kind} {directive.target} "
            f"{directive.name} {options}"
        )
    if not lines:
        lines.append("- (none)")
    return lines


def _sorted_directives(
    directives: Iterable[TemplateDirective],
) -> list[TemplateDirective]:
    return sorted(
        directives,
        key=lambda directive: (
            directive.kind,
            directive.target,
            directive.name,
            directive.source,
            directive.line or 0,
        ),
    )


def _slug_for_template_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    while relative.suffix:
        relative = relative.with_suffix("")
    return relative.as_posix()


def _normalise_option_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {
            key: _normalise_option_value(inner_value)
            for key, inner_value in value.items()
        }
    return value
