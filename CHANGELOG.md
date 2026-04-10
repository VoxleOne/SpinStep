# Changelog

All notable changes to this project are documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and uses [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.5.0a0] – 2026-04-10

### Added
- **Control subsystem** (`spinstep.control`):
  - `OrientationState` — observer-centered spherical state (quaternion + distance)
  - `ControlCommand` — angular + radial velocity commands
  - `ProportionalOrientationController` — P-controller with rate limiting
  - `PIDOrientationController` — PID controller with anti-windup
  - `OrientationTrajectory`, `TrajectoryInterpolator`, `TrajectoryController` —
    waypoint-based trajectory tracking with SLERP interpolation
  - `integrate_state()`, `compute_orientation_error()` — state utilities
- **Math library** (`spinstep.math`):
  - `core` — `quaternion_multiply`, `quaternion_conjugate`, `quaternion_normalize`,
    `quaternion_inverse`
  - `interpolation` — `slerp` (spherical linear), `squad` (spherical cubic)
  - `geometry` — `quaternion_distance`, `rotate_quaternion`,
    `forward_vector_from_quaternion`, `direction_to_quaternion`,
    `angle_between_directions`, `is_within_angle_threshold`
  - `conversions` — `quaternion_from_euler`, `rotation_matrix_to_quaternion`,
    `quaternion_from_rotvec`, `quaternion_to_rotvec`
  - `analysis` — `batch_quaternion_angle`, `angular_velocity_from_quaternions`,
    `get_relative_spin`, `get_unique_relative_spins`, `NodeProtocol`
  - `constraints` — `clamp_rotation_angle`
- `NodeProtocol` — `typing.Protocol` for any object with `.orientation` attribute,
  replacing `object` type hints in `get_relative_spin()` and
  `get_unique_relative_spins()`
- `Node.add_child()` method for ergonomic tree building
- Comprehensive API stability tests (`test_api.py`) — 92+ parametrized tests
  covering all subpackage exports and backward compatibility
- `Typing :: Typed` classifier in `pyproject.toml`
- `Changelog` and `Documentation` project URLs
- `[[tool.mypy.overrides]]` for `scipy.*`, `sklearn.*`, `healpy.*`, `cupy.*`
- CI workflow now triggers on `feature/*` branches

### Changed
- **Traversal classes moved** to `spinstep.traversal` subpackage
  (`node.py`, `continuous.py`, `discrete.py`, `discrete_iterator.py`).
  Top-level imports via `from spinstep import Node` remain backward-compatible.
- **`spinstep.utils` is now a backward-compatible re-export layer**.
  All 13 quaternion functions now delegate to `spinstep.math` — the single
  source of truth.  Direct `from spinstep.utils import …` continues to work.
- Version bumped to `0.5.0a0`

### Deprecated
- Direct imports from `spinstep.utils.quaternion_utils` and
  `spinstep.utils.quaternion_math`.  Use `spinstep.math` instead.

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
