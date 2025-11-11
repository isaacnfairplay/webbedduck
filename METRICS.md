Metrics & Policy
This repo uses multiple signals, not a single score.
What we measure
	•	Cyclomatic complexity (per function, totals) — how many independent paths.
	•	Maintainability Index (per file, average) — coarse maintainability proxy.
	•	Halstead (bugs/volume/length/vocabulary) — size/effort proxies.
	•	Raw counts — SLOC/LLOC, function & expression counts.
	•	Indirection signals (per function) — is_wrapper, wrapper_kind, call_count, distinct_callees, max_attribute_chain, max_call_nesting, max_nesting_depth.
	•	Hotspots — Git churn × complexity per file:
	◦	churn.commits and churn.loc_changed (added + deleted over a window)
	◦	hotspot_score = commits × (cyclomatic_total + 1)
Indirection signals flag wrapper growth and deep attribute chains. Hotspots highlight where refactors pay off most.
Default policy thresholds
	•	Cyclomatic complexity (per function)
Aim < 10; allow up to 15 with justification; avoid > 20.
	•	Maintainability Index (Visual Studio scale)
≥20 (green), 10–19 (yellow), <10 (red). Treat as advisory only.
	•	Hotspot target when touched
Reduce Σ hotspot_score in changed hotspot files by ≥15% (CI gate).
	•	Wrappers
Encourage only when adding seams, removing duplication, or clarifying boundaries.
Why this policy
	•	Capping cyclomatic complexity around 10–15 is widely recommended for testability and risk control (NIST/McCabe).
	•	Cognitive Complexity favors nesting/flow readability; we approximate with max_nesting_depth and call nesting.
	•	Maintainability Index is useful but imperfect; don’t optimize it in isolation.
	•	Hotspots combine churn with complexity to prioritize work that helps the team most.
Limits & cautions
	•	Metrics are proxies. Don’t game them. Prefer refactors that reduce branching, trim nesting, or simplify data flow.
	•	Use your judgment; document exceptions briefly in the PR.
