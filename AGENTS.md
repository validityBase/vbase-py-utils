# AGENTS.md

Primary instructions are in [CLAUDE.md](CLAUDE.md) - read that first.

## Key Pointers

- Persistent agent memory: [internal/agents/memory/MEMORY.md](internal/agents/memory/MEMORY.md)
- GitHub Actions spec: [internal/specs/github-actions.md](internal/specs/github-actions.md)
- Python dependency hashes: [internal/specs/python-dependency-hashes.md](internal/specs/python-dependency-hashes.md)

## Agent Notes

- Keep code, dependency, and workflow changes scoped.
- Update internal specs or memory when CI behavior, dependency policy, or
  repository conventions change.
- Run relevant validation, or list the exact commands and failures when local
  environment limits prevent a clean run.
- Do not commit secrets, private tokens, webhook URLs, generated `.env` files, or
  local virtual environment contents.
