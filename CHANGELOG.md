# Changelog

All notable changes to this project are documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.3.0a0] – 2026-03-26

### Added
- Module-level docstrings to all core and utility modules
- Test suite for `QuaternionDepthIterator` (continuous traversal) — 10 new tests
- Test suite for utility functions (`quaternion_utils`, `quaternion_math`,
  `array_backend`) — 20 new tests
- `healpy` declared as optional dependency in `pyproject.toml`
- PEP 561 `py.typed` marker file for downstream type-checking support
- `__all__` exports in all public modules for explicit API surface control
- GitHub Actions publish workflow for automated PyPI releases
- Release readiness report (`RELEASE_READINESS.md`)

### Fixed
- `DiscreteQuaternionIterator` erroneously multiplied `magnitude()` by 2,
  over-estimating angular distances between orientations
- `QuaternionDepthIterator` called `R.from_quat()` before checking for
  zero-norm child orientations, causing `ValueError` instead of gracefully
  skipping
- `requirements.txt` pinned `scikit-learn>=1.5` while `pyproject.toml` pinned
  `>=1.2` — now both use `>=1.2`
- README stated "Python 3.8+" but `pyproject.toml` requires `>=3.9` — README
  now states "Python 3.9+"
- README project structure listed `setup.py` / `setup.cfg` which no longer
  exist

### Changed
- Version bumped to `0.3.0a0` (PEP 440 pre-release alpha)
- Restructured branch strategy: `main`, `dev`, `features/cuda`
- Clarified setup instructions in `README.md`

---

## [0.1.0] – 2025-05-14

### Added
- Initial codebase and Git setup
- Created branches: `main`, `dev`, and `feature/cuda`
- Basic Python project scaffold
