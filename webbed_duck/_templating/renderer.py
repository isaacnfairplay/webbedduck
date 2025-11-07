"""Rendering logic for ``{{ ctx.* }}`` placeholders."""

from __future__ import annotations

import ast
import re
from typing import Any, Callable, Dict, Mapping

from .errors import TemplateApplicationError
from .formatters import (
    date_offset as _date_offset,
    format_date as _format_date,
    format_number as _format_number,
    format_timestamp as _format_timestamp,
    stringify,
)
from .state import prepare_context

__all__ = ["TemplateRenderer"]


class TemplateRenderer:
    """Render ``{{ ctx.* }}`` expressions against a request context."""

    _placeholder = re.compile(r"{{\s*(.*?)\s*}}")

    def __init__(self, request_context: Mapping[str, Any]):
        self._prepared = prepare_context(request_context)
        self._root = {
            "constants": self._prepared.constants,
            "parameters": self._prepared.parameters,
        }
        self._modifier_handlers: Dict[str, Callable[[Any, list[Any], dict[str, Any]], Any]] = {
            "date_offset": self._handle_date_offset,
            "date_format": self._handle_date_format,
            "timestamp_format": self._handle_timestamp_format,
            "number_format": self._handle_number_format,
        }

    def render(self, template: str) -> str:
        def replace(match: re.Match[str]) -> str:
            expression = match.group(1)
            value = self._evaluate_expression(expression)
            return stringify(value)

        return self._placeholder.sub(replace, template)

    def _evaluate_expression(self, expression: str) -> Any:
        segments = [segment.strip() for segment in expression.split("|")]
        if not segments:
            raise TemplateApplicationError("Empty template expression")

        base, *modifier_segments = segments
        value = self._resolve_path(base)
        filtered_modifiers = [segment for segment in modifier_segments if segment]
        for modifier in filtered_modifiers:
            value = self._apply_modifier(value, modifier)
        return value

    def _resolve_path(self, path_expression: str) -> Any:
        try:
            node = ast.parse(path_expression, mode="eval").body
        except SyntaxError as exc:  # pragma: no cover - delegated to tests
            raise TemplateApplicationError(
                f"Invalid path expression '{path_expression}'"
            ) from exc
        return self._eval_ast(node)

    def _eval_ast(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Name):
            if node.id != "ctx":
                raise TemplateApplicationError(
                    "Only the 'ctx' root is accessible in templates"
                )
            return self._root
        if isinstance(node, ast.Attribute):
            value = self._eval_ast(node.value)
            return self._resolve_getattr(value, node.attr)
        if isinstance(node, ast.Subscript):
            value = self._eval_ast(node.value)
            key = self._eval_ast(node.slice)
            return self._resolve_getitem(value, key)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Index):  # pragma: no cover - Py <3.9 compatibility
            return self._eval_ast(node.value)
        raise TemplateApplicationError("Unsupported expression in template")

    def _resolve_getattr(self, value: Any, attr: str) -> Any:
        if isinstance(value, Mapping) and attr in value:
            return value[attr]
        try:
            return getattr(value, attr)
        except AttributeError as exc:
            raise TemplateApplicationError(
                f"Attribute '{attr}' is not accessible in templates"
            ) from exc
        raise TemplateApplicationError(f"Attribute '{attr}' not found in mapping")

    def _resolve_getitem(self, value: Any, key: Any) -> Any:
        if isinstance(value, Mapping):
            if key in value:
                return value[key]
            raise TemplateApplicationError(
                f"Key '{key}' does not exist for the current mapping"
            )
        raise TemplateApplicationError("Indexed access is only supported for mappings")

    def _apply_modifier(self, value: Any, modifier: str) -> Any:
        try:
            call = ast.parse(modifier, mode="eval").body
        except SyntaxError as exc:  # pragma: no cover - validated in tests
            raise TemplateApplicationError(f"Invalid modifier '{modifier}'") from exc
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise TemplateApplicationError("Modifiers must be function calls")
        func_name = call.func.id
        args = [self._literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: self._literal_eval(kw.value) for kw in call.keywords}

        if func_name == "coalesce":
            default = args[0] if args else kwargs.get("default")
            return value if value is not None else default

        handler = self._modifier_handlers.get(func_name)
        if handler is None:
            raise TemplateApplicationError(f"Unknown modifier '{func_name}'")
        return handler(value, args, kwargs)

    def _handle_date_offset(
        self, value: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        return _date_offset(value, **kwargs)

    def _handle_date_format(
        self, value: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        format_key = args[0] if args else kwargs.get("format_key", "iso")
        return _format_date(value, format_key, self._prepared.date_formats)

    def _handle_timestamp_format(
        self, value: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        format_key = args[0] if args else kwargs.get("format_key", "iso")
        return _format_timestamp(value, format_key, self._prepared.timestamp_formats)

    def _handle_number_format(
        self, value: Any, args: list[Any], kwargs: dict[str, Any]
    ) -> Any:
        format_key = args[0] if args else kwargs.get("format_key", "decimal")
        return _format_number(value, format_key, self._prepared.number_formats)

    def _literal_eval(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except ValueError as exc:  # pragma: no cover - defensive
            raise TemplateApplicationError("Modifiers accept literal arguments") from exc

