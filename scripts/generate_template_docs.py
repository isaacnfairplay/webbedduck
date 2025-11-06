#!/usr/bin/env python3
"""Generate markdown documentation for templating test cases."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webbed_duck.constants import apply_constants
from examples import (
    BUILTIN_TEMPLATE_CASES,
    EXAMPLE_TEMPLATE_CASES,
    SQL_TEMPLATE_CASE,
    build_report_request_context,
)

OUTPUT_DIR = Path("test_docs")


def iter_cases() -> Iterable:
    for case in EXAMPLE_TEMPLATE_CASES:
        yield case
    for case in BUILTIN_TEMPLATE_CASES:
        yield case
    yield SQL_TEMPLATE_CASE


def render_case_markdown(case, context) -> str:
    rendered = apply_constants(case.template, request_context=context)
    lines = [
        f"# {case.slug.replace('_', ' ').title()}",
        "",
        "```sql",
        rendered,
        "```",
        "",
        "## Template",
        "",
        "```jinja",
        case.template,
        "```",
        "",
        "## Context excerpt",
        "",
    ]
    lines.append("```json")
    lines.append(_format_json(context["constants"]))
    lines.append("```")
    parameters = context.get("parameters")
    if parameters:
        lines.extend(["", "## Parameters", "", "```json", _format_json(parameters), "```"])
    if case.description:
        lines.extend(["", case.description])
    lines.append("")
    return "\n".join(lines)


def _format_json(value: Any) -> str:
    import json

    def default(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(_normalise_for_json(value), default=default, indent=2, sort_keys=True)


def _normalise_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_for_json(inner) for key, inner in value.items()}
    if is_dataclass(value):
        return _normalise_for_json(asdict(value))
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    for case in iter_cases():
        context = build_report_request_context()
        document = render_case_markdown(case, context)
        path = output_dir / f"{case.slug}.md"
        path.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    main()
