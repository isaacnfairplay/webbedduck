# Cache route metadata examples

## catalog/sales_overview

* Template: `catalog/sales_overview.sql`

### Invariants
- region → column=`region_code`, separator=`|`, case_insensitive=true

### Directives
- validation validator parameters.country choices {"values": ["CA", "GB", "US"]}
- inline validator parameters.limit positive {"severity": "error"}
- validation validator parameters.limit range {"max": 500, "min": 1}
- validation whitelist parameters.str allowed {"label": "Allowed countries", "values": ["CA", "US"]}

