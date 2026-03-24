# Repository Master Refactorer Instructions

> **Purpose:** Step-by-step instructions for refactoring SpinStep into a clean,
> production-quality Python library.
> **Constraint:** Preserve all existing functionality and tests.
> **Date:** 2026-03-24

---

## Execution Rules (CRITICAL)

* Operate in **proposal mode by default**
* **Do NOT apply changes unless explicitly instructed**
* Always present:

  1. Current State
  2. Issues
  3. Proposed Changes
* Only move to execution after approval

---

## Guiding Principles

1. **Preserve existing functionality** — all existing tests must pass
2. **Do not weaken or remove tests** to make them pass
3. **Do not over-engineer** — avoid unnecessary abstractions
4. **Prefer clarity over cleverness** — readable code wins
5. **Keep the API small** — the 4-class public API is correct as-is
6. **Incremental changes** — each stage must be independently verifiable
7. **Do not expand scope** beyond what is explicitly defined in each stage

---

## Stage Order (MANDATORY)

Execute strictly in this order:

1. Stage 1 — Packaging
2. Stage 2 — Public API (`__all__`)
3. Stage 3 — Internal State (private attributes)
4. Stage 4 — Type Hints
5. Stage 5 — Naming Consistency
6. Stage 6 — Exceptions
7. Stage 7 — CI/CD

Do not skip or reorder stages.

---

## Stage 1: Packaging & PEP 561 Compliance

### 1.1 Add `py.typed` Marker

Create:

```bash
touch spinstep/py.typed
```

Update `MANIFEST.in`:

```
include spinstep/py.typed
```

---

### 1.2 Verify `pyproject.toml` (PEP 621)

Verify (do not assume correctness):

* `[build-system]` uses `setuptools>=61.0`
* `[project]` metadata is complete
* `requires-python = ">=3.9"`
* Dependencies properly defined
* Optional dependencies exist (`gpu`, `healpy`, `dev`)
* Package discovery excludes non-core directories

If any item is incorrect → fix it.

---

### 1.3 Dev Dependencies

Keep `dev-requirements.txt`, but add header:

```
# Dev dependencies — canonical source is pyproject.toml
# Convenience only: pip install -r dev-requirements.txt
```

---

### 1.4 Verify Distribution Contents

```bash
python -m build --sdist
tar tzf dist/*.tar.gz | head -30
```

Confirm exclusion of:

* benchmark/
* demos/
* examples/
* tests/
* docs/

---

### Verification

```bash
pip install -e ".[dev]"
python -c "import spinstep; print(spinstep.__version__)"
pytest tests/ -v
```

---

## Stage 2: Define Public API (`__all__`)

Add explicit exports to all modules.

### Core modules

```python
__all__ = ["Node"]
__all__ = ["QuaternionDepthIterator"]
__all__ = ["DiscreteOrientationSet"]
__all__ = ["DiscreteQuaternionIterator"]
```

### Utils

```python
__all__: list[str] = []
__all__ = ["get_array_module"]
__all__ = ["batch_quaternion_angle"]
```

(Keep utility exports explicit and minimal)

---

### Verification

```bash
pytest tests/ -v
```

---

## Stage 3: Internal State → Private

Rename:

* `xp` → `_xp`
* `rotvecs` → `_rotvecs`
* `stack` → `_stack`

Update all internal references.

---

### Test Safety Rule

* Do NOT rewrite tests unnecessarily
* Prefer testing via public API
* Only update tests if they rely on internal attributes

---

### Verification

```bash
pytest tests/ -v
```

---

## Stage 4: Type Hints

### Replace `Any` with concrete types

Use:

```python
from types import ModuleType
```

Introduce `Protocol` where needed.

---

### Mypy Config

Add:

```toml
[tool.mypy]
strict = true

[[tool.mypy.overrides]]
module = ["scipy.*", "sklearn.*", "healpy.*", "cupy.*"]
ignore_missing_imports = true
```

---

### Constraint

Do NOT rewrite large portions of code just to satisfy typing.

---

### Verification

```bash
mypy spinstep/
pytest tests/ -v
```

---

## Stage 5: Naming Consistency

Rename:

```python
angle → angle_threshold
```

---

### Scope Control

* Only rename the parameter
* Do NOT redesign the API

---

### Verification

```bash
pytest tests/ -v
```

---

## Stage 6: Exceptions (Optional)

Create `exceptions.py` with:

* `SpinStepError`
* `InvalidQuaternionError`
* `BackendError`

Ensure compatibility:

```python
class InvalidQuaternionError(SpinStepError, ValueError):
```

---

### Constraint

Do NOT change exception behavior beyond type replacement.

---

### Verification

```bash
pytest tests/ -v
```

---

## Stage 7: CI/CD (Only after validation)

### Rule

Do NOT modify CI until:

* Tests pass
* Mypy errors are minimal

---

### Add steps

* mypy
* pytest coverage

---

## Output Requirements (EVERY STEP)

Always return:

### User Summary

* Installable: yes/no
* Importable: yes/no
* Tests: X passed / Y failed
* Status: short sentence

### Technical Sections

1. Current State
2. Issues
3. Proposed Changes
4. (Optional) Applied Changes

---

## Final Constraint

* Do NOT re-evaluate previous stages unless explicitly instructed
* Do NOT introduce new refactoring stages
* Do NOT escalate scope

---

*End of instructions.*
