"""Public API for applying request constants to templates."""

from __future__ import annotations

from typing import Any, Mapping

from ._templating import RequestContextStore, TemplateApplicationError, TemplateRenderer

__all__ = ["apply_constants", "TemplateApplicationError", "set_request_context"]

_store = RequestContextStore()


def set_request_context(context: Mapping[str, Any]) -> None:
    """Update the module-level request context used by :func:`apply_constants`."""

    _store.set(context)


def apply_constants(
    template: str, *, request_context: Mapping[str, Any] | None = None
) -> str:
    """Apply request constants to ``template``.

    Parameters
    ----------
    template:
        Text containing ``{{ ctx.* }}`` placeholders.
    request_context:
        Optional context overriding the module-level request context. Extra
        entries in the context are ignored so callers can provide additional
        information for other systems without affecting template resolution.
    """

    context = request_context if request_context is not None else _store.get()
    renderer = TemplateRenderer(context)
    return renderer.render(template)
