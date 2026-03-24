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
| **Test status** | 46 passed / 0 failed / 5 skipped (4 CUDA, 1 healpy) |
| **CI status** | ✅ Active at `.github/workflows/ci.yml` — tests Python 3.9–3.12 |
| **Linting** | ✅ `ruff check spinstep/` passes with zero warnings |
| **Type checking** | ❌ 623 mypy errors under configured `strict = true` |
| **Overall status** | The library is installable, importable, and fully functional. CI is active. Type checking needs work before production release. See [AUDIT.md](AUDIT.md) for full production-readiness assessment. |

---

## Table of Contents

1. [Current State](#1-current-state)
2. [Known Issues](#2-known-issues)
3. [Suggested Next Steps](#3-suggested-next-steps)

---

## 1. Current State

### Overview

SpinStep is an alpha-stage Python library that implements quaternion-driven tree
traversal.  Instead of traversing by position or order, SpinStep uses 3D
orientation (quaternion rotation) to decide which branches to explore.  The core
library is functional, well-documented, and has no import-time side effects.

### Repository Layout

| Directory | Role | Distributed |
|-----------|------|:-----------:|
| `spinstep/` | Core library (≈ 725 LOC across 9 files) | ✅ |
| `tests/` | Unit & integration tests (51 tests across 4 files) | ❌ |
| `docs/` | Markdown documentation (16 files + assets) | ❌ |
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
* **Distribution:** `MANIFEST.in` and `[tool.setuptools.packages.find]` both
  exclude benchmarks, demos, examples, tests, and docs from sdist/wheel.
* **Build:** Both `python -m build --sdist` and `python -m build --wheel` succeed.
* **Install:** `pip install .` and `pip install -e .` work correctly.

### Test Suite

* **Framework:** pytest (51 tests across 4 files in `tests/`).
* **Results:** 46 passed, 0 failed, 5 skipped (4 CuPy/CUDA, 1 healpy).
* **Covered:** `Node`, `QuaternionDepthIterator` (continuous traversal),
  `DiscreteOrientationSet` (init, factories, queries, GPU path),
  `DiscreteQuaternionIterator` (creation, traversal, depth limits), integration
  pipeline, utility functions (`quaternion_utils`, `quaternion_math`,
  `array_backend`).
* **Conditionally covered:** `get_unique_relative_spins` (guarded by
  `pytest.importorskip("healpy")`).

### Code Quality

| Metric | Status |
|--------|--------|
| Type hints | Present on all public functions/methods |
| Docstrings | NumPy-style on all public classes and functions |
| Side effects at import | None (lazy imports for cupy, sklearn, healpy) |
| Linting | `ruff check spinstep/` — zero warnings |
| Type checking | `mypy --strict` configured but 623 errors remain |

### CI/CD

* **Active workflow:** `.github/workflows/ci.yml`
* **Triggers:** push to `main`, `dev`, `features/cuda`; PRs to `main`, `dev`
* **Matrix:** Python 3.9, 3.10, 3.11, 3.12 on `ubuntu-latest`
* **Steps:** `pip install .[dev]` → `ruff check spinstep/` → `pytest tests/`
* **Permissions:** `contents: read` (least privilege)
* **Not included:** mypy, code coverage, release/publish workflow

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

## 2. Known Issues

| # | Severity | Description |
|---|----------|-------------|
| 1 | **Critical** | 623 mypy errors under configured `strict = true` — mainly missing stubs for scipy/sklearn and overly broad `ArrayLike` parameter types. |
| 2 | **Critical** | No `spinstep/py.typed` PEP 561 marker — downstream type checkers won't detect type information. |
| 3 | **High** | CI does not run `mypy` — type regressions are undetected. |
| 4 | **High** | No code coverage reporting in CI. |
| 5 | **Medium** | `CONTRIBUTING.md` references `flake8` instead of the project's actual linter `ruff`. |
| 6 | **Medium** | Missing PyPI metadata: `keywords`, `Homepage`, `Documentation`, `Bug Tracker` URLs. |
| 7 | **Medium** | No release/publish workflow for PyPI distribution. |
| 8 | **Medium** | No `SECURITY.md` at root for GitHub vulnerability reporting. |
| 9 | **Low** | `dev-requirements.txt` duplicates `pyproject.toml [dev]` extras — single source of truth preferred. |
| 10 | **Low** | README contains broken link (`VoxLeone` vs `VoxleOne`) and unclosed HTML attribute on line 78. |

---

## 3. Suggested Next Steps

See [AUDIT.md](AUDIT.md) for a detailed production-readiness audit with a
prioritised remediation roadmap.

**Summary of phases:**

1. **Phase 1 (Critical):** Fix mypy errors, add `py.typed`, add mypy to CI.
2. **Phase 2 (High):** Add coverage, fix docs references, improve metadata.
3. **Phase 3 (Medium):** Add release workflow, SECURITY.md, dependabot, missing tests.
4. **Phase 4 (Polish):** Fix license headers, add `__all__` to utils, pre-commit hooks.

---

*End of report.*
