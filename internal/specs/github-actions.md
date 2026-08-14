# GitHub Actions

## Policy

- Third-party actions are pinned by full commit SHA for reproducibility.
- Shared vBase-owned actions and reusable workflows use `validityBase/vbase-github-actions` with reviewed release tags such as `@v1`.
- Workflow permissions are declared explicitly and kept minimal.
- Secrets must come from GitHub Secrets or deployment configuration, never from committed files or logs.
- Python version is standardized on 3.11 in CI.
- Dependency layout and lock policy are canonical in
  `internal/specs/python-dependency-hashes.md`.

## Local Validation

Use the same dependency setup as CI:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --require-hashes -r requirements/dev.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m unittest discover -s tests
pylint $(git ls-files '*.py')
```

## Workflows

### `.github/workflows/python-dependency-locks.yml`
- Runs on pull requests, pushes to `main`, and manual `workflow_dispatch`.
- Installs lock tooling through `setup-python-deps@v1`.
- Regenerates dependency locks; the workflow fails if the committed lock files differ.
- Installs development dependencies through `setup-python-deps@v1`.
- Installs the package locally with `python -m pip install --no-deps --no-build-isolation -e .`.
- Runs `python -m pip check`.

### `.github/workflows/run-pylint.yml`
- Runs on pushes to all branches, including branch names containing `/`.
- Delegates to `validityBase/vbase-github-actions/.github/workflows/python-lint.yml@v1`.
- Runs `pylint $(git ls-files '*.py')`.

### `.github/workflows/run-unit-tests.yml`
- Runs on pushes to all branches, including branch names containing `/`.
- Checks out the repository with the pinned `actions/checkout` action.
- Installs development dependencies through `setup-python-deps@v1`.
- Installs the package locally with `python -m pip install --no-deps --no-build-isolation -e .`.
- Runs `python -m unittest discover -s tests`.

### `.github/workflows/repo-backup.yml`
- Runs daily and through manual dispatch to create a full-history git bundle
  backup.
- Delegates to `validityBase/vbase-github-actions/.github/workflows/repo-backup.yml@v1`.
- Uses reviewed moving major tags for validityBase-owned shared workflows so
  centrally reviewed fixes roll forward without per-repository pin updates.
- Requires `VBASE_COMMON_REPO_READ_TOKEN` and
  `VBASE_REPO_BACKUP_SECRETS_TOKEN` GitHub Actions secrets.
- Reads object storage credentials from the `vbase-repo-backups` Bitwarden
  project at runtime; bucket lifecycle and restore-test policy live outside
  this repository.
