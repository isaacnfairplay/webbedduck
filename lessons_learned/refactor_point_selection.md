# Selecting Refactor Points

- Always start with `python tools/complexity_report.py auto` to see the measured hotspots before touching code.
- Target files with the largest hotspot score *only* when you can plausibly reduce their cyclomatic total by at least 15%; otherwise, look for the next candidate.
- Inside the hotspot, search for functions with the highest individual CC values via `radon cc -s -n 20 <file>`—shaving those peaks usually yields the fastest hotspot score drop.
