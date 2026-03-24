# SpinStep — Refactor Agent Instructions

> **Purpose:** Step-by-step instructions for refactoring SpinStep into a clean,
> production-quality Python library.  
> **Constraint:** Preserve all existing functionality and tests.  
> **Date:** 2026-03-24

---

## Guiding Principles

1. **Preserve existing functionality** — all 51 tests must continue to pass
2. **Do not over-engineer** — avoid unnecessary abstractions
3. **Prefer clarity over cleverness** — readable code wins
4. **Keep the API small** — the 4-class public API is correct as-is
5. **Incremental changes** — each stage should be independently verifiable

---

## Stage 1: Packaging & PEP 561 Compliance

### 1.1 Add `py.typed` Marker

Create an empty file at `spinstep/py.typed` (PEP 561 marker):

```bash
touch spinstep/py.typed
```

Update `MANIFEST.in` to include it:

```
include spinstep/py.typed
```

### 1.2 Verify `pyproject.toml` Follows PEP 621

The current `pyproject.toml` is already well-structured. Verify these items:

- [x] `[build-system]` uses `setuptools>=61.0` ✅
- [x] `[project]` has name, version, description, authors ✅
- [x] `requires-python = ">=3.9"` ✅
- [x] Dependencies pinned with minimum versions ✅
- [x] Optional dependencies defined (`gpu`, `healpy`, `dev`) ✅
- [x] `[tool.setuptools.packages.find]` excludes non-core dirs ✅

**No changes needed** to pyproject.toml structure.

### 1.3 Clean Up Redundant Files

`dev-requirements.txt` duplicates the `[project.optional-dependencies.dev]`
section in `pyproject.toml`. **Decision:**

- **Keep** `dev-requirements.txt` as a convenience alias, but add a header
  comment clarifying the canonical source is `pyproject.toml`:

```
# Dev dependencies — canonical source is pyproject.toml [project.optional-dependencies.dev]
# This file exists for convenience: pip install -r dev-requirements.txt
```

### 1.4 Verify Distribution Excludes Non-Core

Run `python -m build --sdist` and inspect the tarball to confirm that
`benchmark/`, `demos/`, `examples/`, `tests/`, and `docs/` are NOT included:

```bash
python -m build --sdist
tar tzf dist/spinstep-0.1.0.tar.gz | head -30
```

**Already correct** — `MANIFEST.in` and `[tool.setuptools.packages.find]` both
exclude them.

### Verification

```bash
pip install -e ".[dev]"
python -c "import spinstep; print(spinstep.__version__)"
python -m pytest tests/ -v
```

---

## Stage 2: Add `__all__` to All Modules

Define explicit public APIs for every module. This improves IDE support,
documentation generators, and makes the public/private boundary clear.

### 2.1 Core Modules

Each core module exports a single class. Add `__all__` after imports:

**`spinstep/node.py`:**
```python
__all__ = ["Node"]
```

**`spinstep/traversal.py`:**
```python
__all__ = ["QuaternionDepthIterator"]
```

**`spinstep/discrete.py`:**
```python
__all__ = ["DiscreteOrientationSet"]
```

**`spinstep/discrete_iterator.py`:**
```python
__all__ = ["DiscreteQuaternionIterator"]
```

### 2.2 Utility Modules

**`spinstep/utils/__init__.py`:**
```python
"""Internal utilities for quaternion math and array backend selection."""

__all__: list[str] = []  # Utilities must be explicitly imported from submodules
```

**`spinstep/utils/array_backend.py`:**
```python
__all__ = ["get_array_module"]
```

**`spinstep/utils/quaternion_math.py`:**
```python
__all__ = ["batch_quaternion_angle"]
```

**`spinstep/utils/quaternion_utils.py`:**
```python
__all__ = [
    "quaternion_from_euler",
    "quaternion_distance",
    "rotate_quaternion",
    "is_within_angle_threshold",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotation_matrix_to_quaternion",
    "get_relative_spin",
    "get_unique_relative_spins",
]
```

### Verification

```bash
python -c "from spinstep.utils.quaternion_utils import *; print(dir())"
python -m pytest tests/ -v
```

---

## Stage 3: Mark Internal State as Private

Prefix internal implementation attributes with `_` to signal they are not part
of the public contract. **This is a breaking change for anyone accessing these
directly** — acceptable at 0.x.

### 3.1 `DiscreteOrientationSet` (discrete.py)

| Current | New | Reason |
|---------|-----|--------|
| `self.xp` | `self._xp` | Backend module is an implementation detail |
| `self.rotvecs` | `self._rotvecs` | Cached rotation vectors are internal |

**Update all references** within `discrete.py` that use `self.xp` → `self._xp`
and `self.rotvecs` → `self._rotvecs`.

**Check:** `discrete_iterator.py` accesses `self.orientation_set` but does NOT
directly access `.xp` or `.rotvecs`, so no changes needed there.

### 3.2 `DiscreteQuaternionIterator` (discrete_iterator.py)

| Current | New | Reason |
|---------|-----|--------|
| `self.stack` | `self._stack` | Traversal state is internal |

**Update all references** within `discrete_iterator.py`.

### 3.3 Update Tests

Tests may reference these attributes. Search for:

```bash
grep -rn "\.xp\b\|\.rotvecs\b\|\.stack\b" tests/
```

Update any test assertions that access these internal attributes to use the new
`_`-prefixed names, or test through the public API instead.

### Verification

```bash
ruff check spinstep/
python -m pytest tests/ -v
```

---

## Stage 4: Improve Type Hints

### 4.1 Replace `Any` with Specific Types

**`spinstep/utils/quaternion_math.py`:**

```python
from types import ModuleType
import numpy as np

def batch_quaternion_angle(
    qs1: np.ndarray,
    qs2: np.ndarray,
    xp: ModuleType,
) -> np.ndarray:
```

**`spinstep/discrete.py`:**

```python
from types import ModuleType

class DiscreteOrientationSet:
    _xp: ModuleType
    orientations: np.ndarray  # Keep public — users query this
    _balltree: object  # or Optional[Any] since BallTree lacks stubs
```

### 4.2 Fix mypy Issues (Non-Stub)

**`spinstep/utils/quaternion_utils.py` lines 103-104:**

The `get_relative_spin(nf, nt)` function accesses `.orientation` on parameters
typed as `object`. Fix by either:

a) Adding a `Protocol`:
```python
from typing import Protocol

class HasOrientation(Protocol):
    orientation: np.ndarray

def get_relative_spin(nf: HasOrientation, nt: HasOrientation) -> np.ndarray:
```

b) Or typing the parameters as `Node`:
```python
from spinstep.node import Node

def get_relative_spin(nf: Node, nt: Node) -> np.ndarray:
```

**Prefer option (a)** — it avoids coupling `quaternion_utils` to `Node`.

### 4.3 Add mypy Configuration for Third-Party Stubs

Add to `pyproject.toml`:

```toml
[tool.mypy]
strict = true

[[tool.mypy.overrides]]
module = [
    "scipy.*",
    "sklearn.*",
    "healpy.*",
    "cupy.*",
]
ignore_missing_imports = true
```

### Verification

```bash
mypy --strict spinstep/ 2>&1 | tail -5
# Goal: reduce from 623 errors to <30
python -m pytest tests/ -v
```

---

## Stage 5: Minor Naming Consistency

### 5.1 Parameter Naming

In `DiscreteOrientationSet.query_within_angle()`, rename the `angle` parameter
to `angle_threshold` for consistency with `DiscreteQuaternionIterator`:

```python
def query_within_angle(self, quat: ArrayLike, angle_threshold: float) -> np.ndarray:
```

**This is a public API change.** Update:
1. The method signature
2. The docstring
3. All internal uses of the parameter
4. Tests that call `query_within_angle()`
5. Demo/example scripts that use this parameter

**Alternative:** If backward compatibility is important, keep both names:

```python
def query_within_angle(
    self,
    quat: ArrayLike,
    angle_threshold: float | None = None,
    *,
    angle: float | None = None,  # deprecated
) -> np.ndarray:
```

**Recommendation for 0.x:** Just rename it — no backward compat needed at alpha.

### Verification

```bash
grep -rn "query_within_angle" tests/ demos/ examples/ benchmark/
# Update all call sites
python -m pytest tests/ -v
```

---

## Stage 6: Error Handling (Optional — Future)

This stage is **optional for 0.1.x** but recommended before 1.0.

### 6.1 Custom Exception Hierarchy

Create `spinstep/exceptions.py`:

```python
"""SpinStep exception hierarchy."""

__all__ = [
    "SpinStepError",
    "InvalidQuaternionError",
    "BackendError",
]


class SpinStepError(Exception):
    """Base exception for all SpinStep errors."""


class InvalidQuaternionError(SpinStepError, ValueError):
    """Raised when a quaternion input is invalid (wrong shape, zero norm)."""


class BackendError(SpinStepError, RuntimeError):
    """Raised when array backend selection fails."""
```

**Note:** Inheriting from both `SpinStepError` and `ValueError` preserves
backward compatibility — existing `except ValueError` blocks will still catch
these errors.

### 6.2 Update Error Raises

Replace `ValueError` raises in `node.py` and `discrete.py` with the new
exception types. Example:

```python
# Before
raise ValueError("Orientation must be a quaternion [x,y,z,w]")

# After
from spinstep.exceptions import InvalidQuaternionError
raise InvalidQuaternionError("Orientation must be a quaternion [x,y,z,w]")
```

Export from `__init__.py`:

```python
from .exceptions import SpinStepError, InvalidQuaternionError, BackendError
```

### Verification

```bash
python -m pytest tests/ -v
# All tests should still pass because new exceptions inherit from ValueError
```

---

## Stage 7: CI/CD Improvements (Optional)

### 7.1 Add mypy to CI

Add a step to `.github/workflows/ci.yml`:

```yaml
      - name: Type check with mypy
        run: |
          mypy spinstep/
```

**Note:** Only add this AFTER Stage 4 reduces mypy errors to near-zero.

### 7.2 Add Coverage Measurement

```yaml
      - name: Test with coverage
        run: |
          pip install pytest-cov
          pytest tests/ --cov=spinstep --cov-report=term-missing
```

---

## Refactoring File Map

This table maps the expected changes per file for the full refactoring:

| File | Stage(s) | Changes |
|------|----------|---------|
| `spinstep/py.typed` | 1 | **CREATE** — empty PEP 561 marker |
| `MANIFEST.in` | 1 | Add `include spinstep/py.typed` |
| `dev-requirements.txt` | 1 | Add header comment about canonical source |
| `spinstep/node.py` | 2 | Add `__all__` |
| `spinstep/traversal.py` | 2 | Add `__all__` |
| `spinstep/discrete.py` | 2, 3, 4 | Add `__all__`, rename `xp`→`_xp`, `rotvecs`→`_rotvecs`, improve types |
| `spinstep/discrete_iterator.py` | 2, 3 | Add `__all__`, rename `stack`→`_stack` |
| `spinstep/utils/__init__.py` | 2 | Add `__all__ = []` |
| `spinstep/utils/array_backend.py` | 2 | Add `__all__` |
| `spinstep/utils/quaternion_math.py` | 2, 4 | Add `__all__`, improve type hints |
| `spinstep/utils/quaternion_utils.py` | 2, 4 | Add `__all__`, fix `get_relative_spin` typing |
| `pyproject.toml` | 4 | Add mypy overrides for third-party stubs |
| `spinstep/exceptions.py` | 6 | **CREATE** — custom exception hierarchy |
| `spinstep/__init__.py` | 6 | Export new exceptions |
| `.github/workflows/ci.yml` | 7 | Add mypy and coverage steps |
| `tests/*.py` | 3, 5 | Update attribute access for renamed privates, parameter names |

---

## Example: Final API Usage

After refactoring, the public API should look like this:

```python
import numpy as np
from spinstep import (
    Node,
    QuaternionDepthIterator,
    DiscreteOrientationSet,
    DiscreteQuaternionIterator,
)

# --- Continuous traversal ---
root = Node("root", [0, 0, 0, 1])
child = Node("child", [0, 0, 0.1, 0.995])
root.children.append(child)

step = [0, 0, 0.05, 0.9987]
for node in QuaternionDepthIterator(root, step):
    print(node.name)

# --- Discrete traversal ---
orientations = DiscreteOrientationSet.from_icosahedron()
print(f"Set has {len(orientations)} orientations")

results = orientations.query_within_angle(
    quat=[0, 0, 0, 1],
    angle=np.pi / 4,  # or angle_threshold after Stage 5
)
print(f"Found {len(results)} orientations within 45°")

root = Node("origin", [0, 0, 0, 1])
for node in DiscreteQuaternionIterator(root, orientations):
    print(node.name)

# --- Utility functions (explicit import) ---
from spinstep.utils.quaternion_utils import quaternion_distance, quaternion_from_euler

q = quaternion_from_euler([0, 0, 90], order="zyx", degrees=True)
dist = quaternion_distance(q, [0, 0, 0, 1])
print(f"Distance from identity: {dist:.4f} rad")
```

---

## Priority Order

For maximum impact with minimum risk, execute stages in this order:

1. **Stage 1** (Packaging) — 10 minutes, no code changes
2. **Stage 2** (`__all__`) — 15 minutes, additive only
3. **Stage 4** (Type hints) — 30 minutes, improves tooling
4. **Stage 3** (Private attrs) — 20 minutes, minor breaking change
5. **Stage 5** (Naming) — 10 minutes, minor breaking change
6. **Stage 6** (Exceptions) — 30 minutes, backward compatible
7. **Stage 7** (CI) — 15 minutes, infrastructure only

**Total estimated effort: ~2-3 hours**

---

*End of refactor instructions.*
