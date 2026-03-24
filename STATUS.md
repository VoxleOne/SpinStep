# SpinStep — Repository Status Report

> **Version:** 0.1.0 (Alpha)  
> **Date:** 2026-03-24  
> **License:** MIT  

---

## User Status Summary

| Metric | Status |
|--------|--------|
| **Installable** | ✅ Yes (`pip install .` and `pip install -e .` both succeed) |
| **Importable** | ✅ Yes (`import spinstep` works, exports 4 public classes) |
| **Core functionality working** | ✅ Yes |
| **Test status** | 46 passed / 0 failed / 4 skipped (CUDA not available) |
| **Overall status** | The library is installable, importable, and fully functional; CI workflow is disabled and needs updating. |

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Issues](#2-issues)
3. [Suggested Changes](#3-suggested-changes)

---

## 1. Current State

### Overview

SpinStep is an alpha-stage Python library that implements quaternion-driven tree
traversal.  Instead of traversing by position or order, SpinStep uses 3D
orientation (quaternion rotation) to decide which branches to explore.  The core
library is functional, well-typed, and has no import-time side effects.

### Repository Layout

| Directory | Role | Distributed |
|-----------|------|:-----------:|
| `spinstep/` | Core library (≈ 680 LOC) | ✅ |
| `tests/` | Unit & integration tests (≈ 390 LOC) | ❌ |
| `docs/` | Markdown documentation (≈ 1 500 LOC) | ❌ |
| `demos/` | Educational scripts (5 files) | ❌ |
| `examples/` | Real-world examples (1 file) | ❌ |
| `benchmark/` | Performance benchmarks & QGNN examples | ❌ |

### Packaging

* **Build system:** `pyproject.toml` (PEP 517 / 621), built with
  `setuptools ≥ 61.0`.
* **Dependencies (core):** `numpy ≥ 1.22`, `scipy ≥ 1.10`,
  `scikit-learn ≥ 1.2`.
* **Optional extras:** `[gpu]` → `cupy-cuda12x`, `[healpy]` → `healpy`,
  `[dev]` → `pytest`, `black`, `ruff`, `mypy`.
* **Distribution:** `MANIFEST.in` excludes benchmarks, demos, examples, tests,
  and docs from the sdist/wheel.  `pyproject.toml` `[tool.setuptools.packages.find]`
  also excludes them.
* **Build:** `python -m build --sdist` succeeds.
* **Install:** `pip install .` or `pip install -e .` works correctly.

### Test Suite

* **Framework:** pytest (50 tests across 4 files in `tests/`).
* **Results:** 46 passed, 0 failed, 4 skipped (CuPy/CUDA not available).
* **Covered:** `Node`, `QuaternionDepthIterator` (continuous traversal),
  `DiscreteOrientationSet` (init, factories, queries, GPU path),
  `DiscreteQuaternionIterator` (creation, traversal, depth limits), integration
  pipeline, utility functions (`quaternion_utils`, `quaternion_math`,
  `array_backend`).
* **Not covered:** `get_unique_relative_spins` (requires optional `healpy`
  dependency).

### Code Quality

| Metric | Status |
|--------|--------|
| Type hints | 100 % (all public functions/methods) |
| Docstrings | ≈ 90 % (module + class + public function docstrings) |
| Side effects at import | None (lazy imports for cupy, sklearn, healpy) |
| Linting config | `ruff` and `black` configured (88-char lines) |
| Type checking config | `mypy` strict mode configured |

### CI/CD

* The CI workflow content is stored in `.github/workflows_disabled`, a single
  file outside the `.github/workflows/` directory.  Because GitHub Actions only
  recognises workflow files inside `.github/workflows/`, no CI runs on push or
  PR.
* No active GitHub Actions workflows exist in `.github/workflows/`.
* The disabled workflow still includes Python 3.8 in its test matrix, which is
  not supported by the project (`requires-python = ">=3.9"`).

### Public API

The library exports four classes via `spinstep/__init__.py`:

| Class | Purpose |
|-------|---------|
| `Node` | Tree node with a quaternion orientation `[x, y, z, w]`, auto-normalised |
| `QuaternionDepthIterator` | Continuous rotation-step depth-first traversal |
| `DiscreteOrientationSet` | Queryable set of discrete quaternion orientations |
| `DiscreteQuaternionIterator` | Discrete rotation-step depth-first traversal |

### Internal Utilities (`spinstep/utils/`)

| Module | Key Functions |
|--------|---------------|
| `array_backend` | `get_array_module(use_cuda)` — returns NumPy or CuPy |
| `quaternion_math` | `batch_quaternion_angle(qs1, qs2, xp)` — pairwise angular distances |
| `quaternion_utils` | `quaternion_from_euler`, `quaternion_distance`, `rotate_quaternion`, `is_within_angle_threshold`, `quaternion_conjugate`, `quaternion_multiply`, `rotation_matrix_to_quaternion`, `get_relative_spin`, `get_unique_relative_spins` (HEALPix) |

---

## 2. Issues

| # | Severity | Description |
|---|----------|-------------|
| 1 | **Medium** | CI workflow is disabled — the workflow YAML lives in `.github/workflows_disabled` instead of `.github/workflows/`, so GitHub Actions does not recognise it.  No automated testing runs on push or PR. |
| 2 | **Medium** | Disabled CI workflow still lists Python 3.8 in its test matrix, which conflicts with `requires-python = ">=3.9"` in `pyproject.toml`. |
| 3 | **Low** | Disabled CI workflow uses `flake8` for linting, but the project is configured for `ruff` and `black` in `pyproject.toml` and `dev-requirements.txt`. |
| 4 | **Low** | `get_unique_relative_spins()` has no test coverage (requires optional `healpy` dependency). |
| 5 | **Low** | `dev-requirements.txt` lists `matplotlib`, `numba`, `cython`, `pycuda` which are not used anywhere in the core library or tests. These are stale entries. |

---

## 3. Suggested Changes

| # | Relates to Issue | Action | Details |
|---|-----------------|--------|---------|
| 1 | Issue 1 | Re-enable CI workflow | Rename `.github/workflows_disabled` to `.github/workflows/ci.yml` so GitHub Actions picks it up as an active workflow. |
| 2 | Issue 2 | Update Python test matrix | Remove `3.8` from the matrix; add `3.12` to match pyproject.toml classifiers. Use `['3.9', '3.10', '3.11', '3.12']`. |
| 3 | Issue 3 | Replace `flake8` with `ruff` in CI | Change the lint step from `flake8` to `ruff check spinstep/` to match the project's configured linter. |
| 4 | Issue 4 | Add conditional `healpy` test | Add a test for `get_unique_relative_spins()` guarded by `pytest.importorskip("healpy")`, following the same pattern used for CuPy tests. |
| 5 | Issue 5 | Clean up `dev-requirements.txt` | Remove unused entries (`matplotlib`, `numba`, `cython`, `pycuda`) or add a comment explaining they are for optional experimentation only. |

---

*End of report.*
