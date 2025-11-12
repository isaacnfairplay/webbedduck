"""Resolution helpers for filtering cache results."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Any, Mapping, TYPE_CHECKING

import pyarrow as pa
import pyarrow.compute as pc

from .cache_support import InvariantFilter

if TYPE_CHECKING:  # pragma: no cover - type-checking imports only
    from .cache import CacheKey


@dataclass(frozen=True)
class CacheFilterPipeline:
    """Apply invariant and constant filters in sequence."""

    invariants: Mapping[str, InvariantFilter]

    def apply(
        self, table: pa.Table, key: "CacheKey", *, include_invariants: bool = True
    ) -> pa.Table:
        filtered = table
        if include_invariants:
            filtered = self._apply_invariants(filtered, key)
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


__all__ = ["CacheFilterPipeline"]

