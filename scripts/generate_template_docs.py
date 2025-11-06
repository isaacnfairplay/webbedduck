#!/usr/bin/env python3
"""Generate markdown documentation for templating test cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webbed_duck.constants import apply_constants
from webbed_duck.examples import (
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
    constants = context["constants"]
    if "number" in constants:
        constants["number"] = dict(constants["number"])
    if "timestamp" in constants:
        constants["timestamp"] = dict(constants["timestamp"])
    if "date" in constants:
        constants["date"] = dict(constants["date"])
    lines.append("```json")
    lines.append(_format_context(constants))
    lines.append("```")
    if case.description:
        lines.extend(["", case.description])
    lines.append("")
    return "\n".join(lines)


def _format_context(constants):
    import json

    def default(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(constants, default=default, indent=2, sort_keys=True)


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
