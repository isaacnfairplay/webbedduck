"""Error types used by the templating helpers."""


class TemplateApplicationError(RuntimeError):
    """Raised when a template cannot be rendered safely."""

    __slots__ = ()
