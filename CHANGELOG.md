---

# Changelog

All notable changes to this project are documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and uses [Semantic Versioning](https://semver.org/).

---

## \[Unreleased]

### Added

* MIT license/authorship headers to all `.py` files (\[features/cuda])
* CUDA feature branch introduced to explore GPU acceleration

### Changed

* Restructured repository branching strategy: `main`, `v2.0`, `features/cuda`
* Clarified and updated setup instructions in `README.md`

---

## \[0.2.0] – 2025-06-09

### Changed

* **Core architecture redesign**: Generalized from quaternion iteration on a unit sphere to support **multi-layered concentric spherical structures**
* Refactored quaternion stepper logic to accommodate arbitrary radial layering
* Enhanced modularity and parameterization for reuse across various spherical geometries
* Improved documentation: `README.md` now reflects the conceptual shift and new usage examples

### Removed

* Previous assumptions tied to a single-sphere model to allow for broader application

---

## \[0.1.0] – 2025-05-14

### Added

* Initial codebase and Git repository setup
* Created foundational branches: `main`, `dev`, `feature/cuda`
* Basic Python scaffolding and project structure

---
