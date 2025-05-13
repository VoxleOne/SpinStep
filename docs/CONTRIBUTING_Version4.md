# Contributing to SpinStep

Thank you for your interest in contributing to SpinStep!  
We welcome pull requests, issues, and suggestions.

## How to Contribute

### 1. Issues & Feature Requests
- Please check for existing issues before opening a new one.
- When reporting bugs, include steps to reproduce and relevant logs/traces.
- For feature requests, describe your use case and desired API.

### 2. Pull Requests

- Fork the repository and create a new branch for your patch or feature.
- Follow code style and naming conventions (see below).
- Add or update tests as appropriate.
- Update documentation if your change affects the public API or user experience.
- Reference any related issues in your PR description.

### 3. Code Style

- Write clear, readable Python code.
- Use [PEP8](https://www.python.org/dev/peps/pep-0008/) as a baseline; check with `flake8`.
- Docstrings: Use [NumPy style](https://numpydoc.readthedocs.io/en/latest/format.html) for modules, classes, and functions.
- All public API functions/classes must have docstrings and type annotations.

### 4. Testing

- All new features and bug fixes should come with tests.
- Run all tests with `pytest` before submitting your PR.
- If adding a new traversal mode or orientation set, include edge cases and property-based tests.

### 5. Continuous Integration

- All PRs are automatically tested on multiple Python versions.
- PRs with failing tests or linting will not be merged.

## Project Structure

- `spinstep/` — Core library code
- `tests/` — Unit and integration tests
- `docs/` — Documentation, guides, and usage examples
- `notebooks/` — Interactive demos and educational materials

## Community Standards

- Be respectful and constructive in all discussions.
- We follow the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

Thank you for helping make SpinStep better!