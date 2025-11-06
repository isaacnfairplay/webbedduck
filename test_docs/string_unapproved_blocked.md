# String Unapproved Blocked

```text
TemplateApplicationError: String constant 'unapproved' is not present in the whitelist
```

## Template

```jinja
{{ ctx.constants.str.unapproved }}
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
    "source_path": "/srv/data/reports",
    "unapproved": "DROP TABLE"
  },
  "timestamp": {
    "created": "2024-01-31T12:15:33+00:00"
  }
}
```

Fails because the value for `str.unapproved` is not included in the whitelist (`{"source_path", "report_name"}`).
