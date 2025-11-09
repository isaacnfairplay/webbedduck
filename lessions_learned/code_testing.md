# Code Testing Lessons

- Run `pytest -q` before committing to catch regressions introduced by cache refactors.
- Use property-based samples when validating invariant filters to cover edge-normalisation paths.
- Record flaky cases with seed values so they can be replayed deterministically.
