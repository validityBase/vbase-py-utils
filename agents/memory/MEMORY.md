# Agent Memory

## GitHub Actions
- Third-party GitHub Actions are pinned to full commit SHAs.
- vBase-owned shared actions and reusable workflows use reviewed `validityBase/vbase-github-actions` version tags.
- Pylint delegates to `validityBase/vbase-github-actions/.github/workflows/python-lint.yml@v1`.
- Unit tests use `validityBase/vbase-github-actions/.github/actions/setup-python-deps@v1`.
- Both workflows install `requirements-dev.txt`, which includes `requirements.txt`.
- Push branch filters use `"**"` so branches containing `/` are included.
