# CLAUDE.md

This file is the minimal shared entry point for agentic work in this repository.

## Core Standards

- This repository is a Python 3.11 utility package. Keep changes small,
  tested, and consistent with the existing `unittest`, pylint, and Black style.
- Runtime dependency inputs live in `requirements.in`; generated runtime locks
  live in `requirements.txt`.
- Development dependency inputs live in `requirements-dev.in`; generated
  development locks live in `requirements-dev.txt`.
- Install generated locks with `python -m pip install --require-hashes`.
- `setup.py` reads `requirements.in` into `install_requires`, so runtime
  dependency changes affect downstream package consumers.
- Do not commit secrets, private tokens, generated `.env` files, or local virtual
  environment contents.
- Public documentation belongs in `README.md` or `docs/` if a published docs
  tree is added later. Internal specs, guides, and agent memory belong under
  `internal/`.

## Internal Documentation

- Persistent agent memory: [internal/agents/memory/MEMORY.md](internal/agents/memory/MEMORY.md)
- GitHub Actions spec: [internal/specs/github-actions.md](internal/specs/github-actions.md)
- Dependency hashes: [internal/specs/python-dependency-hashes.md](internal/specs/python-dependency-hashes.md)
