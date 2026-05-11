# GitHub Actions

## Policy

- Third-party actions are pinned by full commit SHA for reproducibility.
- Shared vBase-owned actions and reusable workflows use `validityBase/vbase-github-actions` with reviewed release tags such as `@v1`.
- Workflow permissions are declared explicitly and kept minimal.
- Secrets must come from GitHub Secrets or deployment configuration, never from committed files or logs.
- Workflows install `requirements-dev.txt`, which includes runtime dependencies
  from `requirements.txt`.
- Python version is standardized on 3.11 in CI.

## Local Validation

Use the same dependency file as CI:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python -m unittest discover -s tests
pylint $(git ls-files '*.py')
```

## Workflows

### `.github/workflows/run-pylint.yml`
- Runs on pushes to all branches, including branch names containing `/`.
- Delegates to `validityBase/vbase-github-actions/.github/workflows/python-lint.yml@v1`.
- Installs `requirements-dev.txt` with Python 3.11.
- Runs `pylint $(git ls-files '*.py')`.

### `.github/workflows/run-unit-tests.yml`
- Runs on pushes to all branches, including branch names containing `/`.
- Checks out the repository with the pinned `actions/checkout` action.
- Installs `requirements-dev.txt` with Python 3.11 through `setup-python-deps@v1`.
- Runs `python -m unittest discover -s tests`.
