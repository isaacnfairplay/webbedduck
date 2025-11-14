# Code Refactoring Lessons

- Break cache orchestration into testable helpers to cap cyclomatic complexity.
- Introduce guard clauses for empty datasets before performing costly Arrow materialisation.
- Prefer pure functions for token normalisation to simplify unit testing.
- Use the churn-weighted hotspot report to identify refactor targets so that each reduction meaningfully improves the cache mod
  ules most likely to regress.
