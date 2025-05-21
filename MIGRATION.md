# Migration from Hatch to UV

This document outlines the changes made to migrate the Strands Agents SDK from using Hatch to UV for development workflow management.

## Changes Made

### 1. Added Python Scripts for Development Tasks

Created Python scripts in the `scripts/` directory to replace hatch commands:
  ```
  scripts/
  ├── format.py     # Runs ruff format
  ├── lint.py       # Runs ruff check and mypy
  ├── test.py       # Runs pytest with coverage
  └── test_integ.py # Runs integration tests
  ```

### 2. Updated .pre-commit-config.yaml

- Changed all hooks to use the Python scripts:
  ```yaml
  - id: uv-format
    name: Format code
    entry: python scripts/format.py
    language: system
    pass_filenames: false
    types: [python]
    stages: [pre-commit]
  ```

### 3. Updated GitHub Workflows

- Changed the GitHub workflow to install and use uv instead of hatch:
  ```yaml
  - name: Install uv
    run: |
      pip install uv
  - name: Install dependencies
    run: |
      uv venv
      source .venv/bin/activate || . .venv/Scripts/activate
      uv pip install -e ".[dev,anthropic,litellm,llamaapi,ollama]" --prerelease=allow
  - name: Run Unit tests
    id: tests
    run: |
      source .venv/bin/activate || . .venv/Scripts/activate
      python scripts/test.py
  ```

### 4. Updated Documentation

- Updated CONTRIBUTING.md to reflect the use of uv for development tasks
- Updated README.md to show uv commands for development

### 5. Fixed mypy Configuration

- Updated mypy configuration in pyproject.toml:
  - Set `ignore_missing_imports = true` to handle external dependencies
  - Added specific overrides for modules like anthropic, ollama, etc.
  - Removed the deprecated `follow_untyped_imports` option
  - Updated mypy to version 1.8.0 to fix compatibility issues with Python 3.13

## Benefits of Using UV

1. **Faster dependency resolution**: UV is significantly faster than pip and other package managers
2. **Better virtual environment management**: UV automatically manages virtual environments
3. **Simplified workflow**: UV provides a more intuitive interface for common development tasks
4. **Improved developer experience**: UV makes it easier to debug lint and test errors
5. **Full pyproject.toml support**: UV fully manages and respects pyproject.toml configurations

## Migration Guide for Contributors

If you have an existing clone of the repository that was using hatch:

1. Install uv:
   ```bash
   pip install uv
   ```

2. Create a new virtual environment with all dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]" --prerelease=allow
   ```

3. Update your workflow to use the Python scripts:
   - Instead of `hatch fmt --formatter`, use `python scripts/format.py`
   - Instead of `hatch fmt --linter`, use `python scripts/lint.py`
   - Instead of `hatch test`, use `python scripts/test.py`
   - Instead of `hatch run test-integ`, use `python scripts/test_integ.py`

4. Update pre-commit hooks:
   ```bash
   pre-commit install -t pre-commit -t commit-msg
   ```
