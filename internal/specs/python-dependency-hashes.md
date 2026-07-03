# Python Dependency Hashes

`vbase-py-utils` is an intermediate utility package installed by downstream
repositories. Published package dependencies must stay abstract and
resolver-friendly. CI, tests, linting, and lock-generation tooling are terminal
environments owned by this repository, so those installs use pip hash-checking
mode for reproducibility.

Lock files are generated with Python 3.11 for CI parity.

## Files

- `requirements.in` is the human-edited published runtime dependency source.
  It is read by `setup.py`, must use dependency ranges rather than hash-locked
  pins, and is included in source distributions through `MANIFEST.in`.
- `requirements/src/dev.in` is the human-edited development/test/lint
  environment input. It includes `../../requirements.in` so CI validates the
  package's runtime dependency ranges in a terminal environment.
- `requirements/lock/dev.txt` is generated from `requirements/src/dev.in` and
  includes runtime plus development/test/lint dependencies with hashes.
- `requirements/src/tools.in` is the human-edited lock-regeneration tooling
  input.
- `requirements/lock/tools.txt` is generated from `requirements/src/tools.in`
  and includes the minimal `pip-tools` environment with hashes.

Do not create a generated base/runtime lock for package metadata. Do not edit
generated `.txt` lock files by hand.

## Developer Workflow

Install pinned lock-generation tooling from the minimal lock before running
`pip-compile`. Do not bootstrap with an unpinned `pip install pip-tools`,
because a different `pip-tools` version can produce a different lockfile than
CI.

```bash
python -m pip install --require-hashes -r requirements/lock/tools.txt
```

To add or update a published runtime dependency:

```bash
# edit requirements.in
pip-compile --strip-extras --no-annotate --allow-unsafe --generate-hashes -o requirements/lock/dev.txt requirements/src/dev.in
```

To add or update a development/test/lint dependency:

```bash
# edit requirements/src/dev.in
pip-compile --strip-extras --no-annotate --allow-unsafe --generate-hashes -o requirements/lock/dev.txt requirements/src/dev.in
```

To update the lock-generation tooling, edit the pinned `pip-tools==...`
constraint in `requirements/src/tools.in`, then regenerate
`requirements/lock/tools.txt`.

```bash
# edit the pip-tools==... pin in requirements/src/tools.in
pip-compile --strip-extras --no-annotate --allow-unsafe --generate-hashes -o requirements/lock/tools.txt requirements/src/tools.in
```

Install local development dependencies from the generated lock file:

```bash
python -m pip install --require-hashes -r requirements/lock/dev.txt
python -m pip install --no-deps --no-build-isolation -e .
```

## CI Contract

`.github/workflows/python-dependency-locks.yml` installs the minimal
lock-generation tooling lock with `require-hashes: true`, regenerates terminal
environment lock files, fails if generated files differ from committed files,
installs the development lock, installs package metadata without resolving
dependencies again, and runs `python -m pip check`.

The test and lint workflows install `requirements/lock/dev.txt` with
`require-hashes: true` and install the editable package with:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

`--no-deps --no-build-isolation` is intentional: third-party runtime and
build-time dependencies must come from the committed hashed terminal lock, not
from a second package resolution during the editable install.
