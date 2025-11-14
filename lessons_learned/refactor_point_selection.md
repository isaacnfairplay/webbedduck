# Refactor Point Selection Notes

- Run `python tools/complexity_report.py --base origin/main --churn-since-days 180 auto` (or the closest equivalent when remotes
  are unavailable) before touching code so hotspot data is fresh.
- Sort `reports/base/metrics.json` by `hotspot_score` to find files with both churn and complexity pressure; they offer the best
  return for small diffs.
- Within a hotspot, focus on functions whose cyclomatic complexity is above the surrounding baseline—splitting repeated logic
  into cohesive helpers can drop the reported cost without adding indirection for its own sake.
- Prefer helpers that encode a domain rule (e.g., cache constant preparation) so later features reuse them instead of repeating
  inline comprehensions.
