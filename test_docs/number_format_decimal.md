# Number Format Decimal

```sql
12,456.000
```

## Template

```jinja
{{ ctx.constants.number.visitors | number_format('decimal') }}
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

Applies the 'decimal' number formatter.
