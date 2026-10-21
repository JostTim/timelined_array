# AGENTS.md

## Commands

- Run tests with `uv run pytest tests`
- Run tests, including coverage report: `uv run pytest` (automatically includes `pytest-cov`, `pytest-ipython`)
- Lint with `uv run ruff check .`
- Type check with `uv run ty check .`
- Command order matters: lint → type check → test

## Architecture

Single-package Python library for time-indexed NumPy arrays (timelined_arrays). Extends `np.ndarray` with time-based indexing. Key classes live in `src/timelined_array/` with minimal top-level entrypoints.

## Testing

- Tests live in `tests/` and include pytest fixtures (mock data, time index generation) and parameterized time-slice scenarios.
- Test execution uses pytest-ipython for interactive debugging on failure.
- All timelined operations have direct unit/test coverage with explicit time-slice assertions.

