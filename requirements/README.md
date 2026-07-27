# Python Requirements

Published package dependencies live in `requirements.in` as abstract ranges.
Do not generate a hash-locked runtime requirements file for package metadata.

Human-edited terminal environment inputs live in `requirements/*.in`.
Generated hash-locked terminal environment files live in `requirements/*.txt`.

Do not edit generated `.txt` files by hand. Regenerate them with the documented
commands in `internal/specs/python-dependency-hashes.md`.
