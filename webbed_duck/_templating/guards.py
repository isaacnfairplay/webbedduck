"""Guard evaluation helpers for parameter binding."""

from __future__ import annotations

import datetime as _dt
import operator
import re
from collections.abc import Iterable, Mapping as MappingABC
from contextlib import suppress
from typing import Any, Callable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from .binding import ResolvedParameter

__all__ = ["validate_parameter_guards"]


_GuardPredicate = Callable[[Any], bool] | type | tuple[type, ...]


def _guard_option(
    name: str,
    guards: Mapping[str, Any],
    key: str,
    *,
    predicate: _GuardPredicate | None = None,
    type_label: str,
    transform: Callable[[Any], Any] | None = None,
) -> tuple[Any | None, list[str]]:
    value = guards.get(key)
    if value is None:
        return None, []
    if predicate is not None:
        check = (
            isinstance(value, predicate)
            if isinstance(predicate, (tuple, type))
            else predicate(value)
        )
        if not check:
            return None, [f"Parameter '{name}' {key} guard must be {type_label}"]
    return (transform(value) if transform else value), []


_SIMPLE_GUARDS: Mapping[
    str,
    tuple[
        _GuardPredicate | None,
        str,
        Callable[[Any], Any] | None,
        Callable[[str, Any, Any], Iterable[str]],
    ],
] = {
    "choices": (
        lambda candidate: isinstance(candidate, Iterable)
        and not isinstance(candidate, (str, bytes)),
        "iterable",
        list,
        lambda name, current, options: []
        if current in options
        else [f"Parameter '{name}' must be one of: {', '.join(map(str, options))}"],
    ),
    "regex": (
        str,
        "a string pattern",
        None,
        lambda name, current, pattern: (
            [f"Parameter '{name}' must be a string to apply regex guard"]
            if not isinstance(current, str)
            else (
                []
                if re.fullmatch(pattern, current)
                else [f"Parameter '{name}' must match pattern '{pattern}'"]
            )
        ),
    ),
}


def _iter_simple_guard_errors(name: str, value: Any, guards: Mapping[str, Any]) -> Iterable[str]:
    for key, (predicate, type_label, transform, validator) in _SIMPLE_GUARDS.items():
        option, type_errors = _guard_option(
            name,
            guards,
            key,
            predicate=predicate,
            type_label=type_label,
            transform=transform,
        )
        if type_errors:
            yield from type_errors
        elif option is not None:
            yield from validator(name, value, option)


_BOUND_CHECKS: Mapping[str, tuple[Callable[[Any, Any], bool], str]] = {
    "min": (operator.lt, "at least"),
    "max": (operator.gt, "at most"),
}


def _coerce_guard_values(
    name: str,
    guard_key: str,
    bounds: Mapping[str, Any],
    converter: Callable[[Any], float | int | None],
) -> tuple[dict[str, tuple[Any, float | int]], list[str]]:
    numeric: dict[str, tuple[Any, float | int]] = {}
    errors: list[str] = []
    for label in ("min", "max"):
        raw = bounds.get(label)
        if raw is None:
            continue
        coerced = converter(raw)
        if coerced is None:
            errors.append(
                f"Parameter '{name}' {guard_key} guard '{label}' must be numeric"
            )
        else:
            numeric[label] = (raw, coerced)
    return numeric, errors


def _validate_length_guard(
    name: str,
    value: Any,
    bounds: Mapping[str, Any],
    _resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    try:
        length = len(value)  # type: ignore[arg-type]
    except TypeError:
        return [f"Parameter '{name}' length guard requires a sized value"]

    numeric, errors = _coerce_guard_values(name, "length", bounds, _as_int)
    return errors or [
        f"Parameter '{name}' length must be {_BOUND_CHECKS[label][1]} {raw}"
        for label, (raw, coerced) in numeric.items()
        if _BOUND_CHECKS[label][0](length, coerced)
    ]


def _range_violation_messages(
    name: str, numeric_value: float | int, numeric: Mapping[str, tuple[Any, float | int]]
) -> list[str]:
    active = [
        (label, raw)
        for label, (raw, coerced) in numeric.items()
        if _BOUND_CHECKS[label][0](numeric_value, coerced)
    ]
    if not active:
        return []
    if {"min", "max"}.issubset(numeric):
        min_raw, _ = numeric["min"]
        max_raw, _ = numeric["max"]
        return [f"Parameter '{name}' must be between {min_raw} and {max_raw}"]
    label, raw = active[0]
    return [f"Parameter '{name}' must be {_BOUND_CHECKS[label][1]} {raw}"]


def _validate_range_guard(
    name: str,
    value: Any,
    bounds: Mapping[str, Any],
    _resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    if not any(bounds.get(label) is not None for label in ("min", "max")):
        return [f"Parameter '{name}' range guard requires 'min' or 'max'"]
    numeric_value = _as_float(value)
    if numeric_value is None:
        return [f"Parameter '{name}' must be numeric to apply range guard"]

    numeric, errors = _coerce_guard_values(name, "range", bounds, _as_float)
    return errors or _range_violation_messages(name, numeric_value, numeric)


def _validate_datetime_window_guard(
    name: str,
    value: Any,
    window: Mapping[str, Any],
    _resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    earliest = window.get("earliest")
    latest = window.get("latest")
    if earliest is None and latest is None:
        return [
            f"Parameter '{name}' datetime_window guard requires 'earliest' or 'latest'",
        ]

    value_dt, value_error = _parse_datetime_for_guard(value, name, for_value=True)
    if value_error is not None:
        return [value_error]

    def _check_boundary(
        label: str,
        raw_boundary: Any,
        comparator: Callable[[Any, Any], bool],
        descriptor: str,
    ) -> str | None:
        if raw_boundary is None:
            return None
        boundary_dt, boundary_error = _parse_datetime_for_guard(
            raw_boundary, name, boundary=label
        )
        if boundary_error is not None:
            return boundary_error
        if value_dt is None or boundary_dt is None:
            return None
        if _datetimes_mixed_timezone_awareness(value_dt, boundary_dt):
            return (
                f"Parameter '{name}' datetime_window guard requires value and {label}"
                " boundary to use the same timezone awareness"
            )
        if comparator(value_dt, boundary_dt):
            formatted = (
                raw_boundary.isoformat()
                if isinstance(raw_boundary, _dt.datetime)
                else str(raw_boundary)
            )
            return f"Parameter '{name}' must not be {descriptor} {formatted}"
        return None

    return [
        error
        for error in (
            _check_boundary("earliest", earliest, operator.lt, "earlier than"),
            _check_boundary("latest", latest, operator.gt, "later than"),
        )
        if error is not None
    ]


def _resolve_compare_configuration(
    name: str,
    compare: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> tuple[str | None, "_CompareOperator" | None, "ResolvedParameter" | None, list[str]]:
    target_name = compare.get("parameter")
    if not isinstance(target_name, str):
        return (
            None,
            None,
            None,
            [f"Parameter '{name}' compare guard requires a parameter name"],
        )

    operator_name = compare.get("operator") or "eq"
    if not isinstance(operator_name, str):
        return (
            target_name,
            None,
            None,
            [f"Parameter '{name}' compare guard operator must be a string"],
        )

    comparator = _COMPARE_OPERATORS.get(operator_name)
    if comparator is None:
        return (
            target_name,
            None,
            None,
            [
                f"Parameter '{name}' compare guard uses unsupported operator '{operator_name}'"
            ],
        )

    other = resolved.get(target_name)
    if other is None:
        return (
            target_name,
            comparator,
            None,
            [
                f"Parameter '{name}' compare guard requires parameter '{target_name}' to be resolved first"
            ],
        )

    return target_name, comparator, other, []


def _validate_compare_guard(
    name: str,
    value: Any,
    compare: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    target_name, comparator, other, config_errors = _resolve_compare_configuration(
        name, compare, resolved
    )
    if config_errors:
        return config_errors

    with suppress(TypeError):
        assert comparator is not None and other is not None
        if comparator.compare(value, other.value):
            return []

        message = compare.get("message")
        if isinstance(message, str):
            return [message]
        if message is None:
            return [comparator.message(name, target_name)]
        return [str(message)]

    return [
        f"Parameter '{name}' compare guard could not compare with parameter '{target_name}'",
    ]


_MAPPING_GUARDS: Mapping[
    str,
    Callable[[str, Any, Mapping[str, Any], Mapping[str, "ResolvedParameter"]], list[str]],
] = {
    "length": _validate_length_guard,
    "range": _validate_range_guard,
    "datetime_window": _validate_datetime_window_guard,
    "compare": _validate_compare_guard,
}


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime_value(raw: Any) -> _dt.datetime | None:
    if raw is None:
        return None
    if isinstance(raw, _dt.datetime):
        return raw
    if isinstance(raw, str):
        with suppress(ValueError):
            return _dt.datetime.fromisoformat(raw)
    return None


def _datetime_guard_error(name: str, *, for_value: bool, boundary: str | None) -> str:
    if for_value:
        return (
            f"Parameter '{name}' must be a datetime or ISO 8601 string to apply datetime_window guard"
        )
    qualifier = f" {boundary}" if boundary else ""
    return (
        f"Parameter '{name}' datetime_window{qualifier} must be a datetime or ISO 8601 string"
    )


def _parse_datetime_for_guard(
    raw: Any, name: str, *, for_value: bool = False, boundary: str | None = None
) -> tuple[_dt.datetime | None, str | None]:
    parsed = _coerce_datetime_value(raw)
    if parsed is not None:
        return parsed, None
    if raw is None:
        return None, None
    return None, _datetime_guard_error(name, for_value=for_value, boundary=boundary)


def _datetimes_mixed_timezone_awareness(
    left: _dt.datetime | None, right: _dt.datetime | None
) -> bool:
    if left is None or right is None:
        return False
    return (left.tzinfo is None) != (right.tzinfo is None)


class _CompareOperator:
    __slots__ = ("_func", "_description")

    def __init__(self, func, description: str) -> None:
        self._func = func
        self._description = description

    def compare(self, left: Any, right: Any) -> bool:
        return self._func(left, right)

    def message(self, name: str, other: str) -> str:
        if self._description == "different from":
            return f"Parameter '{name}' must be different from parameter '{other}'"
        return (
            f"Parameter '{name}' must be {self._description} parameter '{other}'"
        )


_COMPARE_OPERATORS: Mapping[str, _CompareOperator] = {
    "eq": _CompareOperator(operator.eq, "equal to"),
    "ne": _CompareOperator(operator.ne, "different from"),
    "lt": _CompareOperator(operator.lt, "less than"),
    "lte": _CompareOperator(operator.le, "less than or equal to"),
    "gt": _CompareOperator(operator.gt, "greater than"),
    "gte": _CompareOperator(operator.ge, "greater than or equal to"),
}


def validate_parameter_guards(
    name: str,
    value: Any,
    guards: Mapping[str, Any],
    resolved: Mapping[str, "ResolvedParameter"],
) -> list[str]:
    """Return guard violation messages for ``name`` given ``value``."""

    if not guards:
        return []

    errors = list(_iter_simple_guard_errors(name, value, guards))
    for key, handler in _MAPPING_GUARDS.items():
        mapping, type_errors = _guard_option(
            name,
            guards,
            key,
            predicate=MappingABC,
            type_label="a mapping",
        )
        if type_errors:
            errors.extend(type_errors)
        elif mapping is not None:
            errors.extend(handler(name, value, mapping, resolved))
    return errors

