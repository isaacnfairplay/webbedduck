"""Support utilities shared across cache components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

NULL_SENTINEL = "__null__"


def decode_null(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value == NULL_SENTINEL:
        return None
    return value


def freeze_str_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value for key, value in mapping.items()})


def freeze_token_mapping(
    mapping: Mapping[str, Iterable[str]] | Mapping[str, Sequence[str]]
) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({str(name): tuple(tokens) for name, tokens in mapping.items()})


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)


def normalize_items(mapping: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not mapping:
        return tuple()
    return tuple(
        (
            str(key),
            NULL_SENTINEL if (value := mapping[key]) is None else stringify(value),
        )
        for key in sorted(mapping)
    )


def decode_ttl_seconds(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def ensure_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


@dataclass(frozen=True)
class InvariantFilter:
    """Normalisation and predicate configuration for invariant constants."""

    key: str
    column: str | None = None
    separator: str = "|"
    case_insensitive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "column", self.column or self.key)

    def normalize(self, value: Any) -> tuple[str, ...]:
        if value is None:
            return (NULL_SENTINEL,)
        candidates: Iterable[Any]
        if isinstance(value, str):
            candidates = value.split(self.separator)
        else:
            candidates = ensure_sequence(value)
        normalized = tuple(
            token
            for candidate in candidates
            if (token := self._normalize_candidate(candidate)) is not None
        )
        return tuple(sorted(dict.fromkeys(normalized)))

    def _normalize_candidate(self, candidate: Any) -> str | None:
        if candidate is None:
            return NULL_SENTINEL
        token = str(candidate).strip()
        if not token:
            return None
        if self.case_insensitive:
            return token.lower()
        return token

    def encode(self, tokens: tuple[str, ...]) -> str:
        if not tokens:
            return ""
        return self.separator.join(tokens)

    def apply(self, table: pa.Table, tokens: tuple[str, ...]) -> pa.Table:
        predicate = self._predicate_for(table, tokens)
        if predicate is None:
            return table
        if isinstance(predicate, pa.ChunkedArray):
            predicate = predicate.combine_chunks()
        return table.filter(predicate)

    def _predicate_for(
        self, table: pa.Table, tokens: tuple[str, ...]
    ) -> pa.Array | pa.ChunkedArray | None:
        if not tokens:
            return None
        column_name = self.column or self.key
        if column_name not in table.column_names:
            return None
        column_data = table[column_name]
        column_for_compare = (
            pc.utf8_lower(column_data)
            if self.case_insensitive and pa.types.is_string(column_data.type)
            else column_data
        )
        predicate = self._membership_predicate(column_for_compare, tokens)
        return self._include_nulls(predicate, column_data, tokens)

    def _membership_predicate(
        self, column_data: pa.Array | pa.ChunkedArray, tokens: tuple[str, ...]
    ) -> pa.Array | pa.ChunkedArray | None:
        compare_tokens = tuple(token for token in tokens if token != NULL_SENTINEL)
        if not compare_tokens:
            return None
        try:
            token_array = pa.array(compare_tokens, type=column_data.type)
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            token_array = pa.array(compare_tokens).cast(column_data.type)
        return pc.is_in(column_data, value_set=token_array)

    def _include_nulls(
        self,
        predicate: pa.Array | pa.ChunkedArray | None,
        column_data: pa.Array | pa.ChunkedArray,
        tokens: tuple[str, ...],
    ) -> pa.Array | pa.ChunkedArray:
        if NULL_SENTINEL not in tokens:
            return predicate
        null_predicate = pc.is_null(column_data)
        if predicate is None:
            return null_predicate
        return pc.or_(predicate, null_predicate)


def encode_constant_value(
    key: str,
    value: Any,
    filters: Mapping[str, InvariantFilter],
    invariant_tokens: dict[str, tuple[str, ...]],
) -> str:
    filter_config = filters.get(key)
    if filter_config is None:
        return NULL_SENTINEL if value is None else stringify(value)
    tokens = filter_config.normalize(value)
    invariant_tokens[key] = tokens
    return filter_config.encode(tokens)


def compute_digest(
    route_slug: str,
    parameters: tuple[tuple[str, str], ...],
    constants: tuple[tuple[str, str], ...],
) -> str:
    payload = json.dumps(
        {
            "route": route_slug,
            "parameters": parameters,
            "constants": constants,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """Metadata describing a cached materialisation on disk."""

    key: "CacheKey"
    path: Path
    created_at: datetime
    expires_at: datetime
    row_count: int
    page_size: int
    invariants: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class CacheEntryMetadata:
    """Serialisable metadata describing a cache entry on disk."""

    created_at: datetime
    ttl: timedelta
    row_count: int
    page_size: int
    route_slug: str
    parameters: tuple[tuple[str, str], ...]
    constants: tuple[tuple[str, str], ...]
    raw_parameters: Mapping[str, Any] = field(repr=False)
    raw_constants: Mapping[str, Any] = field(repr=False)
    invariants: Mapping[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def for_entry(
        cls,
        *,
        key: "CacheKey",
        table: pa.Table,
        created_at: datetime,
        ttl: timedelta,
        page_size: int,
        invariants: Mapping[str, tuple[str, ...]] | None,
    ) -> "CacheEntryMetadata":
        invariant_tokens = freeze_token_mapping(
            {name: tuple(tokens) for name, tokens in (invariants or {}).items()}
        )
        return cls(
            created_at=created_at,
            ttl=ttl,
            row_count=table.num_rows,
            page_size=page_size,
            route_slug=key.route_slug,
            parameters=tuple(key.parameters),
            constants=tuple(key.constants),
            raw_parameters=freeze_str_mapping(dict(key.parameter_values)),
            raw_constants=freeze_str_mapping(dict(key.constant_values)),
            invariants=invariant_tokens,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CacheEntryMetadata":
        created_at = datetime.fromisoformat(data["created_at"])
        ttl = timedelta(seconds=decode_ttl_seconds(data.get("ttl_seconds", 0)))
        page_source = data.get("page_size") or data.get("row_count", 1) or 1
        page_size = max(1, int(page_source))
        return cls(
            created_at=created_at,
            ttl=ttl,
            row_count=int(data.get("row_count", 0)),
            page_size=page_size,
            route_slug=str(data["route_slug"]),
            parameters=tuple(tuple(item) for item in data.get("parameters", [])),
            constants=tuple(tuple(item) for item in data.get("constants", [])),
            raw_parameters=freeze_str_mapping(dict(data.get("raw_parameters") or {})),
            raw_constants=freeze_str_mapping(dict(data.get("raw_constants") or {})),
            invariants=freeze_token_mapping(dict(data.get("invariants") or {})),
        )

    @property
    def expires_at(self) -> datetime:
        return self.created_at + self.ttl

    def to_cache_entry(self, *, key: "CacheKey", path: Path) -> CacheEntry:
        return CacheEntry(
            key=key,
            path=path,
            created_at=self.created_at,
            expires_at=self.expires_at,
            row_count=self.row_count,
            page_size=self.page_size,
            invariants=self.invariants,
        )

    def to_cache_key(self, *, digest: str) -> "CacheKey":
        from .cache import CacheKey  # local import to avoid circular dependency

        return CacheKey(
            route_slug=self.route_slug,
            parameters=self.parameters,
            constants=self.constants,
            digest=digest,
            _raw_parameters=self.raw_parameters,
            _raw_constants=self.raw_constants,
            _invariant_tokens=self.invariants,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "ttl_seconds": self.ttl.total_seconds(),
            "row_count": self.row_count,
            "page_size": self.page_size,
            "route_slug": self.route_slug,
            "parameters": list(self.parameters),
            "constants": list(self.constants),
            "raw_parameters": dict(self.raw_parameters),
            "raw_constants": dict(self.raw_constants),
            "invariants": {
                name: list(tokens) for name, tokens in self.invariants.items()
            },
        }


def encode_constants(
    constants: Mapping[str, Any],
    filters: Mapping[str, InvariantFilter],
) -> tuple[tuple[str, str], ...]:
    invariant_tokens: dict[str, tuple[str, ...]] = {}
    encoded = tuple(
        (
            name,
            encode_constant_value(name, value, filters, invariant_tokens),
        )
        for name, value in sorted(constants.items())
    )
    return encoded, invariant_tokens


__all__ = [
    "NULL_SENTINEL",
    "CacheEntry",
    "CacheEntryMetadata",
    "InvariantFilter",
    "compute_digest",
    "decode_null",
    "decode_ttl_seconds",
    "encode_constant_value",
    "encode_constants",
    "ensure_sequence",
    "freeze_str_mapping",
    "freeze_token_mapping",
    "normalize_items",
    "stringify",
]

