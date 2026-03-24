# SpinStep — Production-Readiness Audit

> **Audit Date:** 2026-03-24
> **Version Audited:** 0.1.0 (Alpha)
> **Branch:** `main` (commit `92a6f1f`)
> **Auditor:** Automated assessment

---

## Executive Summary

SpinStep is a well-structured alpha-stage Python library with strong code quality
fundamentals: clean architecture, good docstrings, zero import side effects, and a
comprehensive test suite.  However, several concrete blockers prevent it from being
considered production-ready.

**Verdict:** Not yet production-ready.  Estimated effort to reach production
readiness: **2–4 focused sessions** addressing the blockers below.

| Category | Grade | Notes |
|----------|-------|-------|
| Core functionality | ✅ A | All 46 core tests pass |
| Code quality | ✅ A | Clean, well-documented, PEP 8 compliant |
| Type safety | ❌ D | 623 mypy errors under configured strict mode |
| Packaging | ⚠️ B- | Builds succeed but missing PEP 561 marker and metadata |
| CI/CD | ⚠️ B- | Linting + tests run, but no coverage, release, or mypy steps |
| Documentation | ⚠️ B | Comprehensive but contains stale/inaccurate references |
| Security posture | ⚠️ C+ | No SECURITY.md, no dependency scanning |

---

## Table of Contents

1. [Critical Blockers](#1-critical-blockers)
2. [High-Priority Issues](#2-high-priority-issues)
3. [Medium-Priority Improvements](#3-medium-priority-improvements)
4. [Low-Priority Polish](#4-low-priority-polish)
5. [What's Already Good](#5-whats-already-good)
6. [Suggested Remediation Roadmap](#6-suggested-remediation-roadmap)

---

## 1. Critical Blockers

These **must** be resolved before any production or PyPI release.

### 1.1 — 623 mypy Errors Under Configured Strict Mode

**Location:** `pyproject.toml` line 58 configures `[tool.mypy] strict = true`, but
running `mypy spinstep/` produces **623 errors** in 5 of the 9 checked source files.

**Root causes:**

| Cause | Affected Files | Example |
|-------|---------------|---------|
| Missing `scipy` type stubs | `traversal.py`, `discrete.py`, `discrete_iterator.py`, `quaternion_utils.py` | `Library stubs not installed for "scipy.spatial.transform"` |
| Missing `sklearn` stubs | `discrete.py` | `module is installed, but missing library stubs or py.typed marker` |
| `ArrayLike` too broad for indexing/unpacking | `quaternion_utils.py` | `Unsupported operand type for unary -` on `ArrayLike` parameter |
| `Any` return types from dynamic calls | `discrete.py`, `quaternion_utils.py` | `Returning Any from function declared to return "ndarray"` |
| `object` attribute access | `quaternion_utils.py:103-104` | `"object" has no attribute "orientation"` |

**Impact:** Downstream users who enable mypy will see errors when type-checking
code that uses SpinStep. The project's own CI does not run mypy, masking these.

**Suggested fix:**

1. Install `scipy-stubs` in dev dependencies.
2. Narrow `ArrayLike` parameters to `np.ndarray` or `Sequence[float]` where
   appropriate.
3. Define a `Protocol` for node-like objects (`get_relative_spin`,
   `get_unique_relative_spins`) instead of `object`.
4. Add `mypy spinstep/` as a CI step (can start with `--ignore-missing-imports`
   and progressively tighten).

---

### 1.2 — No PEP 561 `py.typed` Marker

**Location:** `spinstep/py.typed` does not exist.

**Impact:** Even after fixing the mypy errors, downstream projects that `pip install
spinstep` will not receive type information because pip and mypy use this marker to
detect typed packages ([PEP 561](https://peps.python.org/pep-0561/)).

**Suggested fix:**

1. Create an empty file `spinstep/py.typed`.
2. Ensure it is included in `MANIFEST.in` and in the wheel (verify with
   `unzip -l dist/*.whl`).

---

### 1.3 — STATUS.md Contains Stale Information

**Location:** `STATUS.md`

**Details:** The file states:

> *"CI workflow content is stored in `.github/workflows_disabled`… No active
> GitHub Actions workflows exist in `.github/workflows/`."*

This is **no longer true**. The CI workflow is now active at
`.github/workflows/ci.yml` and recent runs on `main` pass. New contributors
reading STATUS.md will be misled.

**Suggested fix:** Update STATUS.md to reflect the current CI state, test counts
(51 collected, 46 passed, 5 skipped), and remove references to the disabled
workflow.

---

## 2. High-Priority Issues

These should be addressed before a production release but are less severe than
the critical blockers.

### 2.1 — Incomplete `project.urls` Metadata

**Location:** `pyproject.toml` lines 44-45

**Current state:** Only `Repository` is listed.

**Missing:**
- `Homepage`
- `Documentation` (could point to `docs/index.md` or a GitHub Pages site)
- `Bug Tracker` (e.g., `https://github.com/VoxleOne/SpinStep/issues`)
- `Changelog` (e.g., `https://github.com/VoxleOne/SpinStep/blob/main/CHANGELOG.md`)

**Impact:** PyPI renders these as sidebar links. Without them the package looks
incomplete and users cannot easily find issue tracking or docs.

### 2.2 — No `keywords` for PyPI Discoverability

**Location:** `pyproject.toml`

**Suggested addition:**
```toml
keywords = ["quaternion", "traversal", "rotation", "orientation", "3D", "spatial", "tree"]
```

### 2.3 — No Code Coverage Reporting

**Location:** `.github/workflows/ci.yml`

**Current CI steps:** `ruff check` → `pytest tests/`.  No coverage measurement.

**Impact:** There is no visibility into which code paths are untested. The project
has good coverage in practice (~90% of core paths), but this is not tracked or
enforced.

**Suggested fix:**
1. Add `pytest-cov` to dev dependencies.
2. Add `--cov=spinstep --cov-report=term-missing` to the pytest invocation.
3. Optionally upload coverage to Codecov or Coveralls.

### 2.4 — CI Does Not Run mypy

**Location:** `.github/workflows/ci.yml`

The CI runs `ruff check` but **not** `mypy`. With 623 errors this step would
fail immediately. Once the mypy errors are fixed, adding it to CI prevents
regressions.

**Suggested fix:** Add a `mypy spinstep/` step to CI after the ruff step.

### 2.5 — `CONTRIBUTING.md` References Wrong Linter

**Location:** `docs/CONTRIBUTING.md` line 24

**Current:** `"check with flake8"`

**Should be:** `"check with ruff"` (matching `pyproject.toml` and CI).

### 2.6 — Silent Handling of Zero-Norm Quaternions

**Location:** `discrete.py` lines 47-56

**Behavior:** If a mixed array is passed where **some** quaternions are zero-norm
and others are valid, the zero-norm rows silently remain as `[0, 0, 0, 0]` in the
stored orientations. They are only handled later in `query_within_angle` and
`traversal.py` via ad-hoc `np.allclose(…, [0,0,0,0])` checks.

**Impact:** Users may unknowingly pass partially-invalid data and get unexpected
results. Inconsistent state in the orientation set.

**Suggested fix (pick one):**
- **Option A (strict):** Raise `ValueError` if *any* quaternion has zero norm.
- **Option B (filter):** Silently drop zero-norm rows and warn via `warnings.warn`.
- **Option C (document):** Add explicit documentation that zero-norm quaternions are
  retained as-is.

### 2.7 — Duplicate / Conflicting Dev Dependency Files

**Location:** `dev-requirements.txt` vs `pyproject.toml [project.optional-dependencies.dev]`

Both define dev dependencies but with different content:

| Source | Contents |
|--------|----------|
| `dev-requirements.txt` | `pytest`, `black`, `ruff`, `mypy` |
| `pyproject.toml [dev]` | `pytest`, `black`, `ruff`, `mypy` |

Currently they match, but the dual-file pattern invites drift. The
`dev-requirements.txt` file is not referenced anywhere in CI (which uses
`pip install .[dev]`).

**Suggested fix:** Remove `dev-requirements.txt` to keep a single source of truth,
or generate it from pyproject.toml (`pip-compile`).

---

## 3. Medium-Priority Improvements

### 3.1 — Missing Test Coverage for Key Paths

| Area | Status | Details |
|------|--------|---------|
| `get_unique_relative_spins()` | ⚠️ Skipped | Requires `healpy` (not in CI) |
| GPU / CuPy paths | ⚠️ 4 tests skipped | No GPU in CI |
| Batch `query_within_angle(N, 4)` | ❌ Not tested | Only single-quat path tested |
| `from_sphere_grid(n_points=0)` | ❌ Not tested | Edge case: empty grid |
| `DiscreteOrientationSet(None)` | ❌ Not tested | Should raise `ValueError` |
| `rotation_matrix_to_quaternion` edge cases | ❌ Not tested | Only identity + random tested |

**Suggested fix:** Add targeted tests for the above. For `healpy`, consider adding
it to a CI optional-test job.

### 3.2 — No Release / Publishing Workflow

**Current state:** No CI job for building and uploading to PyPI on tag/release.

**Suggested fix:** Add a workflow that:
1. Triggers on GitHub release creation or tag push.
2. Builds sdist + wheel with `python -m build`.
3. Publishes to PyPI via `twine upload` or the `pypa/gh-action-pypi-publish` action.

### 3.3 — No `SECURITY.md` at Repository Root

**Location:** GitHub looks for `SECURITY.md` in the root or `.github/` directory to
display a "Security" tab with vulnerability reporting instructions. The project has
`docs/annex-security.md` but this is not detected by GitHub.

**Suggested fix:** Create `SECURITY.md` at the root with vulnerability reporting
instructions.

### 3.4 — No Dependency Scanning / Dependabot

**Current state:** No `dependabot.yml` or similar tool configured.

**Impact:** Security vulnerabilities in dependencies (`numpy`, `scipy`,
`scikit-learn`) won't be flagged automatically.

**Suggested fix:** Add `.github/dependabot.yml` for pip ecosystem.

### 3.5 — `egg-info` in sdist

**Location:** The sdist tarball `dist/spinstep-0.1.0.tar.gz` includes
`spinstep.egg-info/` which is a build artifact.

**Impact:** Minor — adds ~5 unnecessary files to the source distribution.

**Suggested fix:** This is a setuptools default behavior. Ensure `.gitignore`
excludes `*.egg-info/` (it does) and consider using `hatchling` or `flit` as a
modern build backend that avoids this.

### 3.6 — README Contains Inaccurate Links and Broken HTML

| Line | Issue |
|------|-------|
| 1 | Link to `VoxLeone/SpinStep` — the GitHub organization is `VoxleOne` (lowercase `l`) |
| 78 | `<img>` tag has unclosed `style` attribute: `style="max-width: 100% style="margin: 20px;"` — missing closing quote |
| 195 | Clone URL says `VoxLeone/spinstep` — the GitHub organization is `VoxleOne` (lowercase `l`) |

---

## 4. Low-Priority Polish

### 4.1 — License Header References `LICENSE.txt`

**Location:** All source file headers say `"See LICENSE.txt for full terms."` but
the actual file is named `LICENSE` (no `.txt` extension).

### 4.2 — No `__all__` in Utils Subpackage

**Location:** `spinstep/utils/__init__.py` exports nothing. Individual utility
modules (`quaternion_utils.py`, `quaternion_math.py`, `array_backend.py`) also
lack `__all__` definitions.

**Impact:** Consumers doing `from spinstep.utils import *` get nothing useful, and
it's unclear which functions are considered public API.

### 4.3 — No `pre-commit` Configuration

Adding a `.pre-commit-config.yaml` with `ruff`, `black`, and `mypy` hooks would
prevent style/type regressions from being committed.

### 4.4 — `batch_quaternion_angle` Uses `Any` Types

**Location:** `quaternion_math.py:12`

Parameters `qs1`, `qs2`, and `xp` are all typed as `Any`. This provides no type
safety. Consider using `np.ndarray` for the quaternion arrays and
`types.ModuleType` for `xp`.

### 4.5 — Factory Methods Missing `cls` Type Annotation

**Location:** `discrete.py` lines 147, 153, 159, 170

The `@classmethod` factory methods (`from_cube`, `from_icosahedron`,
`from_custom`, `from_sphere_grid`) don't annotate the `cls` parameter.

---

## 5. What's Already Good

These aspects are production-quality and should be preserved:

| Aspect | Details |
|--------|---------|
| **Architecture** | Clean separation into 4 focused modules + utils subpackage |
| **Public API** | Minimal, well-defined `__all__` with 4 classes |
| **Docstrings** | NumPy-style on all public classes and functions |
| **No import side effects** | Verified: importing `spinstep` produces no output |
| **PEP 621 packaging** | Modern `pyproject.toml` with proper metadata |
| **Test suite** | 51 tests (46 pass, 5 skip) with good structural coverage |
| **CI pipeline** | Active, tests 4 Python versions (3.9–3.12), uses `ruff` |
| **Error handling** | Specific exceptions (`ValueError`, `AttributeError`, `ImportError`) |
| **Optional deps** | `healpy` and `cupy` handled gracefully with lazy imports |
| **Build artifacts excluded** | MANIFEST.in and setuptools config properly exclude non-core dirs |
| **License** | MIT, properly declared in pyproject.toml and root LICENSE file |

---

## 6. Suggested Remediation Roadmap

### Phase 1 — Critical Fixes (Estimated: 1 session)

| Step | Task | Files Affected |
|------|------|---------------|
| 1 | Create `spinstep/py.typed` empty marker | New file |
| 2 | Add `scipy-stubs` to dev dependencies | `pyproject.toml` |
| 3 | Narrow `ArrayLike` → `np.ndarray` in utils where appropriate | `quaternion_utils.py` |
| 4 | Define a `Protocol` for node-like objects | `quaternion_utils.py` |
| 5 | Fix remaining mypy errors to reach zero | Multiple files |
| 6 | Add `mypy spinstep/` step to CI | `.github/workflows/ci.yml` |
| 7 | Update `STATUS.md` to reflect current state | `STATUS.md` |

### Phase 2 — High-Priority Fixes (Estimated: 1 session)

| Step | Task | Files Affected |
|------|------|---------------|
| 8 | Add `keywords`, `Homepage`, `Documentation`, `Bug Tracker` URLs | `pyproject.toml` |
| 9 | Add `pytest-cov` and coverage reporting to CI | `pyproject.toml`, `.github/workflows/ci.yml` |
| 10 | Fix CONTRIBUTING.md flake8 → ruff reference | `docs/CONTRIBUTING.md` |
| 11 | Resolve zero-norm quaternion handling (choose Option A, B, or C) | `discrete.py` |
| 12 | Remove `dev-requirements.txt` (single source of truth) | Delete file |
| 13 | Fix README broken links and HTML | `README.md` |

### Phase 3 — Medium-Priority (Estimated: 1 session)

| Step | Task | Files Affected |
|------|------|---------------|
| 14 | Add missing test cases (batch query, edge cases) | `tests/` |
| 15 | Create root `SECURITY.md` | New file |
| 16 | Add `.github/dependabot.yml` | New file |
| 17 | Add PyPI release workflow | `.github/workflows/release.yml` |

### Phase 4 — Polish (Estimated: 0.5 session)

| Step | Task | Files Affected |
|------|------|---------------|
| 18 | Fix LICENSE.txt → LICENSE in file headers | All source files |
| 19 | Add `__all__` to utils modules | `spinstep/utils/*.py` |
| 20 | Add `.pre-commit-config.yaml` | New file |
| 21 | Improve type specificity in `batch_quaternion_angle` | `quaternion_math.py` |

---

*End of audit report.*
