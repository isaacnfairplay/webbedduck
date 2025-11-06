"""Parameter helpers for templating configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Generic, Hashable, Iterable, TypeVar

from .errors import TemplateApplicationError

__all__ = ["ParameterWhitelist", "StringParameterWhitelist"]

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class ParameterWhitelist(Generic[T]):
    """Bundle pairing requested parameter keys with an allowed policy."""

    requested: Iterable[T]
    allowed: Iterable[T]
    label: str = "Parameter whitelist"

    def __post_init__(self) -> None:
        requested = _freeze_iterable(self.requested, "Parameter whitelist must be iterable")
        allowed = _freeze_iterable(
            self.allowed, "Allowed parameter whitelist keys must be iterable"
        )
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "allowed", allowed)

    def resolve(self) -> FrozenSet[T]:
        """Return the validated set of requested keys."""

        disallowed = self.requested - self.allowed
        if disallowed:
            keys = ", ".join(sorted(map(str, disallowed)))
            raise TemplateApplicationError(
                f"{self.label} contains keys outside the allowed set: {keys}"
            )
        return self.requested


def StringParameterWhitelist(
    *, requested: Iterable[str], allowed: Iterable[str]
) -> ParameterWhitelist[str]:
    """Helper constructing a string-specific whitelist bundle."""

    return ParameterWhitelist[str](
        requested=requested, allowed=allowed, label="String whitelist"
    )


def _freeze_iterable(value: Iterable[T], error_message: str) -> FrozenSet[T]:
    if isinstance(value, str):
        raise TemplateApplicationError(error_message)
    try:
        return frozenset(value)
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise TemplateApplicationError(error_message) from exc
