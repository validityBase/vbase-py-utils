# GitHub Actions

## Policy

- Third-party actions are pinned by full commit SHA for reproducibility.
- Shared vBase-owned actions and reusable workflows use `validityBase/vbase-github-actions` with reviewed release tags such as `@v1`.
- Workflow permissions are declared explicitly and kept minimal.
- Secrets must come from GitHub Secrets or deployment configuration, never from committed files or logs.
- Workflows install generated hash-locked requirements with `require-hashes`.
- Python version is standardized on 3.11 in CI.

## Local Validation

Use the same dependency file as CI:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --require-hashes -r requirements/lock/dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest discover -s tests
pylint $(git ls-files '*.py')
```

## Workflows

### `.github/workflows/python-dependency-locks.yml`
- Runs on pull requests, pushes to `main`, and manual `workflow_dispatch`.
- Installs `requirements/lock/tools.txt` through `setup-python-deps@v1` with Python 3.11 and `require-hashes: true`.
- Regenerates `requirements/lock/tools.txt` and `requirements/lock/dev.txt`; the workflow fails if the committed lock files differ.
- Installs `requirements/lock/dev.txt` through `setup-python-deps@v1` with Python 3.11 and `require-hashes: true`.
- Installs the package locally with `python -m pip install --no-deps --no-build-isolation -e .`.
- Runs `python -m pip check`.

### `.github/workflows/run-pylint.yml`
- Runs on pushes to all branches, including branch names containing `/`.
- Delegates to `validityBase/vbase-github-actions/.github/workflows/python-lint.yml@v1`.
- Installs `requirements/lock/dev.txt` with Python 3.11 and `require-hashes: true`.
- Runs `pylint $(git ls-files '*.py')`.

### `.github/workflows/run-unit-tests.yml`
- Runs on pushes to all branches, including branch names containing `/`.
- Checks out the repository with the pinned `actions/checkout` action.
- Installs `requirements/lock/dev.txt` with Python 3.11 through `setup-python-deps@v1` and `require-hashes: true`.
- Installs the package locally with `python -m pip install --no-deps --no-build-isolation -e .`.
- Runs `python -m unittest discover -s tests`.
