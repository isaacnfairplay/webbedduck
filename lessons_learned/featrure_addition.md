# Feature Addition Lessons

- Thread invariant metadata through new routes from day one to avoid cache key churn later.
- Validate storage format compatibility when adding exporters to prevent partial feature rollouts.
- Surface feature flags via dependency injection so tests can exercise both code paths cheaply.
