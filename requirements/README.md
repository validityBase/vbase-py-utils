# Python Requirements

Published package dependencies live in `../requirements.in` as abstract ranges.
Do not generate a hash-locked runtime requirements file for package metadata.

Human-edited terminal environment inputs live in `src/`. Generated hash-locked
terminal environment files live in `lock/`.

Do not edit files in `lock/` by hand. Regenerate them with the documented
commands in `../internal/specs/python-dependency-hashes.md`.
