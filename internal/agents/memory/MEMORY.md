# Agent Memory

## Repository Purpose

`vbase-py-utils` contains shared Python utility modules used by validityBase
projects.

## Documentation Layout

- Root `CLAUDE.md` and `AGENTS.md` stay small and point to internal docs.
- Persistent memory lives in `internal/agents/memory/MEMORY.md`.
- Internal specs and maintenance notes live under `internal/specs/`.
- Public documentation belongs in `README.md` or `docs/` if a published docs tree
  is added later.

## Dependency Notes

- Runtime dependency inputs live in `requirements.in`; generated runtime locks
  live in `requirements.txt`.
- Development dependency inputs live in `requirements-dev.in`; generated
  development locks live in `requirements-dev.txt`.
- Install generated lock files with `python -m pip install --require-hashes -r <file>`.
- `setup.py` reads `requirements.in` into `install_requires`; runtime dependency
  changes affect package consumers.
- `requirements-dev.in` includes `requirements.in`.
- Use the pinned `pip-tools` from `requirements-dev.txt` before regenerating
  lock files.
- `tqdm` is a direct runtime dependency used by `vbase_utils.sim` for optional
  progress bars.

## GitHub Actions

- Third-party GitHub Actions are pinned to full commit SHAs.
- vBase-owned shared actions and reusable workflows use reviewed `validityBase/vbase-github-actions` version tags.
- Pylint delegates to `validityBase/vbase-github-actions/.github/workflows/python-lint.yml@v1`.
- Unit tests use `validityBase/vbase-github-actions/.github/actions/setup-python-deps@v1`.
- Both workflows install `requirements-dev.txt` with `require-hashes`.
- `python-dependency-locks.yml` verifies lock freshness, hash installs,
  editable install, and `pip check`.
- Push branch filters use `"**"` so branches containing `/` are included.
