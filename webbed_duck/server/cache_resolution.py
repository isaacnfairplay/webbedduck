"""Resolution helpers for filtering cache results and superset detection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from typing import Any, Iterable, Mapping, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from .cache_support import CacheEntry, InvariantFilter

if TYPE_CHECKING:  # pragma: no cover - type-checking imports only
    from .cache import CacheKey, CacheStorage


@dataclass(frozen=True)
class CacheFilterPipeline:
    """Apply invariant and constant filters in sequence."""

    invariants: Mapping[str, InvariantFilter]

    def apply(self, table: pa.Table, key: "CacheKey") -> pa.Table:
        filtered = self._apply_invariants(table, key)
        return self._apply_constants(filtered, key)

    def _apply_invariants(self, table: pa.Table, key: "CacheKey") -> pa.Table:
        filtered = table
        for name, tokens in key.invariant_tokens.items():
            filter_config = self.invariants.get(name)
            if filter_config is None:
                continue
            filtered = filter_config.apply(filtered, tokens)
        return filtered

    def _apply_constants(self, table: pa.Table, key: "CacheKey") -> pa.Table:
        constant_items = tuple(
            (column, target)
            for column, target in key.constant_values.items()
            if column not in self.invariants
        )
        if not constant_items:
            return table
        return reduce(self._filter_by_column, constant_items, table)

    def _filter_by_column(
        self, table: pa.Table, column_target: tuple[str, Any]
    ) -> pa.Table:
        column, target = column_target
        if column not in table.column_names:
            return table
        predicate = self._constant_predicate(table[column], target)
        if isinstance(predicate, pa.ChunkedArray):
            predicate = predicate.combine_chunks()
        return table.filter(predicate)

    def _constant_predicate(
        self, column_data: pa.Array | pa.ChunkedArray, target: Any
    ) -> pa.Array | pa.ChunkedArray:
        if target is None:
            return pc.is_null(column_data)
        try:
            scalar = pa.scalar(target, type=column_data.type)
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            scalar = pa.scalar(target)
        return pc.equal(column_data, scalar)


@dataclass(frozen=True)
class _RequestedKey:
    """Pre-normalised view of the requested cache key for superset matching."""

    digest: str
    parameters: dict[str, Any]
    constants: dict[str, Any]
    invariants: dict[str, set[str]]

    @classmethod
    def from_cache_key(cls, key: "CacheKey", invariant_names: Iterable[str]) -> "_RequestedKey":
        invariant_set = {
            name: set(tokens)
            for name, tokens in key.invariant_tokens.items()
        }
        return cls(
            digest=key.digest,
            parameters=dict(key.parameter_values),
            constants=_filter_non_invariant_constants(
                dict(key.constant_values), invariant_names
            ),
            invariants=invariant_set,
        )


def _filter_non_invariant_constants(
    constants: Mapping[str, Any], invariant_names: Iterable[str]
) -> dict[str, Any]:
    invariant_set = set(invariant_names)
    return {
        name: value
        for name, value in constants.items()
        if name not in invariant_set
    }


@dataclass(frozen=True)
class SupersetResolver:
    """Locate superset cache entries compatible with a requested key."""

    invariants: Mapping[str, InvariantFilter]

    def find(
        self,
        storage: "CacheStorage",
        key: "CacheKey",
        now: datetime,
    ) -> CacheEntry | None:
        if not key.invariant_tokens:
            return None
        requested = _RequestedKey.from_cache_key(key, self.invariants)
        for entry in storage.scan_entries(route_slug=key.route_slug, now=now):
            if self._is_superset(entry, requested):
                return entry
        return None

    def _is_superset(self, entry: CacheEntry, requested: _RequestedKey) -> bool:
        if entry.key.digest == requested.digest:
            return False
        if dict(entry.key.parameter_values) != requested.parameters:
            return False
        candidate_constants = _filter_non_invariant_constants(
            dict(entry.key.constant_values), self.invariants
        )
        if candidate_constants != requested.constants:
            return False
        entry_invariants = {name: set(tokens) for name, tokens in entry.invariants.items()}
        if entry_invariants.keys() != requested.invariants.keys():
            return False
        return all(
            requested.invariants[name].issubset(entry_invariants[name])
            for name in requested.invariants
        )


__all__ = ["CacheFilterPipeline", "SupersetResolver"]

