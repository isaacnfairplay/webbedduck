SELECT
    order_id,
    country_code,
    total_amount
FROM analytics.order_facts
WHERE country_code IN (
    {{ ctx.parameters.str.country }}
)
LIMIT {{ ctx.parameters.number.limit }}{{ webbed_duck.validator("parameters.limit", "positive", severity="error", hint="Limits must be positive") }}
AND {{ ctx.parameters.number.limit }} <= 1000{{ webbed_duck.validator(target="parameters.limit", name="range", min=1, max=1000) }}
