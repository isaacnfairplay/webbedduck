SELECT
    order_id,
    country,
    total_amount
FROM analytics.daily_sales
WHERE country IN ({{ ctx.parameters.str.allowed_countries | join(", ") }})
LIMIT {{ ctx.parameters.number.limit }}{{ webbed_duck.validator("parameters.limit", "positive", severity="error") }}
