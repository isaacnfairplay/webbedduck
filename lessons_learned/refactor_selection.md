# Refactor Selection Notes

- Ran `python tools/complexity_report.py auto` to identify hotspots.
- Focused on `webbed_duck/server/template_metadata.py` because it held one of the top three hotspot scores and a single function (`_directives_for_spec`) accounted for almost a quarter of the file's cyclomatic complexity.
- Looked for logic that could be expressed declaratively; replacing long conditional chains with data-driven dispatchers provided a measurable way to lower the hotspot score without sprawling edits.
