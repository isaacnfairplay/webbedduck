# Sql Round Trip

```sql
SELECT * FROM '/srv/data/reports/metrics.csv' WHERE run_date = 'January 31, 2024' AND created_at >= '2024-01-31T12:15:33+00:00'
```

## Template

```jinja
SELECT * FROM '{{ ctx.constants.str.source_path }}/metrics.csv' WHERE run_date = '{{ ctx.constants.date.run | date_format('month-name') }}' AND created_at >= '{{ ctx.constants.timestamp.created | timestamp_format('iso') }}'
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

## Parameters

```json
{
  "date": {
    "format": {
      "mm-dd-yy": "%m-%d-%y",
      "month-name": "%B %d, %Y",
      "yyyy-mm-dd": "%Y-%m-%d"
    }
  },
  "number": {
    "format": {
      "decimal": ",.3f",
      "percent": {
        "spec": ".0%"
      }
    }
  },
  "str": {
    "whitelist": {
      "allowed": [
        "report_name",
        "source_path"
      ],
      "label": "String whitelist",
      "requested": [
        "report_name",
        "source_path"
      ]
    }
  },
  "timestamp": {
    "format": {
      "iso": "ISO",
      "unix": "UNIX",
      "unix_ms": "UNIX_MS"
    }
  }
}
```

End-to-end SQL snippet using multiple template segments.
