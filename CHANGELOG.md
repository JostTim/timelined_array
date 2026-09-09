# Changelog

All notable changes made to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.6] - 2026-09-09

This version has breaking changes in relation to the previous 0.x versions (removed legacy namings such as TA_Timeline, etc...)

### Added
- Refactored automated tests into multiple files (suite covering indexing, serialization, edge policies...)
- CI/CD: test runs on push with Codecov coverage reporting
- Ruff and ty for formatting, linting and type checking. I have yet to include them in the badges or to test type hinting or documentation coverage, at publish time.

### Changed
- Migrated project and dependency management from PDM to uv
- As a result, I switched PyPI publishing to `uv build` / `uv publish`
- Enhanced slicing and typing of time synchronization (decoupled from BaseTimeArray)
- Improved type hinting across the package
- Bumped required python version to 3.12. **Expect to see a raise to python 3.13 or 3.14 in the very following versions**

### Fixed
- `EdgePolicy` handling: refined both the implementation and its tests
- `TimeIndexer` typing and iteration issues, including slice and `GeneratorType` annotations rejected at runtime
- `rollaxis` behavior
- Array wrap after deprecation when used without `return_scalar` as positional argument
- Compatibility with numpy 2.*

## [0.1.0] - 2024-06-03

### Added
- Initial public release of `timelined_array`
- `TimelinedArray` class: numpy-like array with a time dimension
- `TimeIndexer`: time-based indexing, boundaries, time-to-index resolution
- Masked array support (`MaskedTimelinedArray`) and ufunc support keeping timelines sane.
- Reductions (`mean`, `std`, ...) with timeline-consistent results

[Unreleased]: https://github.com/JostTim/timelined_array/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JostTim/timelined_array/releases/tag/v0.1.0
