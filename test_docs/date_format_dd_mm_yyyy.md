# Date Format Dd Mm Yyyy

```sql
31-01-2024
```

## Template

```jinja
{{ ctx.constants.date.run | date_format('dd-mm-yyyy') }}
```

## Context excerpt

```json
{
  "date": {
    "run": "2024-01-31"
  },
  "misc": {
    "active": true,
    "optional": null
  },
  "number": {
    "discount": "0.125",
    "visitors": 12456
  },
  "str": {
    "report_name": "Daily Metrics",
    "source_path": "/srv/data/reports"
  },
  "timestamp": {
    "created": "2024-01-31T12:15:33+00:00"
  }
}
```

Applies the 'dd-mm-yyyy' date formatter.
