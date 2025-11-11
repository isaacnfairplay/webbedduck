Agents Playbook
Purpose
Make better refactors by using code metrics wisely. Prefer improvements where they matter (hotspots), avoid gaming metrics, and justify big vs. small changes with data.
Golden Rules
	•	Install deps & run tests before and after changes. Failing tests block the PR.
	•	Use the repository tool of record for complexity: tools/complexity_report.py.
	•	Improve where it counts: prioritize hotspots (files that are both complex and frequently changed).
	•	Refactor size fits context: big refactors on true hotspots; surgical changes elsewhere.
	•	Do not optimize the metric; optimize the code (no metric-gaming).
How to run the tool
# Fast path (auto-detect base, compute churn, write reports/, print summary)
python tools/complexity_report.py

# Explicit base + gates used by CI
python tools/complexity_report.py auto \
  --base origin/main \
  --churn-since-days 180 \
  --gate-hotspot-improvement 0.15 \
  --gate-max-cyc-per-func 20
Outputs:
	•	reports/<label>/metrics.json
	•	reports/<label>/radon_cc.txt
	•	reports/<label>/radon_mi.txt
	•	reports/delta.json
	•	reports/summary.txt (also printed)
Decision Algorithm (English code)
INPUT: your changes, test suite results, reports/ from the tool
LET HOTSPOT(file) = (file.churn.commits over last 180d) × (file.cyclomatic.total + 1)
LET TOP_HOTSPOT_SET = top 20% of files by base HOTSPOT

1) If your PR touches any file in TOP_HOTSPOT_SET:
     TARGET: reduce Σ HOTSPOT across touched hotspot files by ≥15%
     ELSE provide a brief written justification in the PR why a large refactor is deferred.

2) Everywhere:
   - Avoid *introducing* functions with cyclomatic complexity > 20.
   - For any function you touch that has complexity ≥ 15, prefer a ≥10% reduction.
   - Wrappers/indirection:
       OK: test seams, memoization/caching, cross-cutting adapters, removing duplication.
       NOT OK: blind trampoline helpers that only add a hop and obscure control flow.
       Heuristic checks in reports: wrapper_count/ratio, avg_fanout, max_attribute_chain.

3) If a small increase in complexity unlocks a meaningful simplification elsewhere or deletes dead code:
     Allowed, if:
       - tests remain green, AND
       - hotspot Σ improves OR risk decreases (nesting, fan-out, duplication), AND
       - you note the tradeoff in the PR.

4) Re-run the tool; attach `reports/summary.txt` and upload `reports/` as artifact.

5) PR must include the "Complexity Report" section (template below).
PR Template (required section)
### Complexity Report (auto)

Command:
`python tools/complexity_report.py auto --base origin/main --churn-since-days 180`

**Touched hotspots (base top 20%)**: <list>

**Hotspot score (touched hotspots)**
- Before → After (Δ and %): <X → Y (Δ Z, P%)>  ✅/❌ meets 15% target

**Most complex function improved**
- Path / Name:
- Cyclomatic before → after (Δ ≥10%): <…>

**Indirection signals (changed files)**
- Wrapper ratio: <before → after>
- Avg fanout (calls/func): <before → after>
- Max attribute chain: <before → after>

**Notes**
- Test command run + result:
- Rationale for refactor size (big/small):
Scope Clarifications
	•	“Core codebase” excludes tests/, examples/, tools/, and generated code unless changes there are the purpose of the PR.
	•	Metrics guide behavior; they don’t override correctness or maintainability. Prefer readability, fewer branches, less nesting, and reduced coupling.
