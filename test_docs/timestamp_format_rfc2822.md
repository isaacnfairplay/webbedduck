# Timestamp Format Rfc2822

```sql
Wed, 31 Jan 2024 12:15:33 +0000
```

## Template

```jinja
{{ ctx.constants.timestamp.created | timestamp_format('rfc2822') }}
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

Applies the 'rfc2822' timestamp formatter.
