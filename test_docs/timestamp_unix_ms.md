# Timestamp Unix Ms

```sql
1706703333000
```

## Template

```jinja
{{ ctx.constants.timestamp.created | timestamp_format('unix_ms') }}
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

Renders the created timestamp as a UNIX millisecond string.
