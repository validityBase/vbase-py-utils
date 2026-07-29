# validityBase Python Utilities

The package contains common Python utilities used across projects.

## Quickstart Guide

1. Clone the repository:
    ```bash
    git clone https://github.com/validityBase/vbase-py-utils.git
    cd vbase-py-utils
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    python3.11 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    python -m pip install --require-hashes -r requirements/dev.txt
    python -m pip install --no-deps --no-build-isolation -e .
    ```

4. For vBase API access, set up environment variables:
Create a `.env` file in the project root with the following variables:
    ```bash
    # vBase Configuration
    VBASE_API_KEY=your_api_key_here           # API key for vBase authentication
    VBASE_API_URL=your_api_url_here           # vBase API endpoint URL
    VBASE_COMMITMENT_SERVICE_PRIVATE_KEY=your_private_key_here  # Private key for vBase commitment service

    # AWS Configuration
    AWS_ACCESS_KEY_ID=your_aws_access_key     # AWS access key for S3 operations
    AWS_SECRET_ACCESS_KEY=your_aws_secret_key # AWS secret key for S3 operations
    S3_BUCKET=your_bucket_name                # S3 bucket name for storing portfolio data
    S3_FOLDER=your_folder_name                # S3 folder path within the bucket
    ```

5. Run pre-commit hooks and pylint:
   ```bash
   pre-commit run --all-files
   pylint $(git ls-files '*.py')
   ```

## Modules

### `vbase_utils.sim` — Causal time-series simulation

`sim(data, callback, time_index)` drives a forward-only simulation loop that
guarantees the callback never sees data from the future.

At each timestamp `t` in `time_index`:
1. Every object in `data` is **masked** to rows where `index <= t` (or
   `first_level <= t` for MultiIndex inputs — see table below).
2. DataFrame columns that are **entirely NaN** in the masked window are dropped
   (so the callback sees only "live" columns — useful for datasets where new
   features are added over time).
3. `callback(masked_data)` is called with the masked dict.
4. Return values (`pd.Series` or `pd.DataFrame`) are accumulated and concatenated
   into the final result dict.

**Accepted index types**

| Index type | Masking rule |
|---|---|
| `DatetimeIndex` | `index <= t` |
| `MultiIndex` with `DatetimeIndex` as first level | `first_level <= t` (e.g. `(date, symbol)` cross-sectional panels) |

**Example — cross-sectional panel**

```python
from vbase_utils.sim import sim
import pandas as pd
import numpy as np

dates = pd.date_range("2020-01-01", periods=52, freq="W")
symbols = ["AAPL", "MSFT", "GOOG"]
mi = pd.MultiIndex.from_product([dates, symbols], names=["date", "symbol"])
panel = pd.DataFrame({"signal": np.random.randn(len(mi))}, index=mi)

def callback(data):
    t = data["panel"].index.get_level_values("date").max()
    xs = data["panel"][data["panel"].index.get_level_values("date") == t]["signal"]
    xs = xs.copy()
    xs.index = xs.index.get_level_values("symbol")
    return {"signal_at_t": xs}

result = sim({"panel": panel}, callback, dates, progress=True)

# result["signal_at_t"] is a wide (52 × 3) DataFrame — dates × symbols.
# Stack back to (date, symbol) MultiIndex:
long = result["signal_at_t"].stack(dropna=True).rename_axis(["date", "symbol"])
```

## Updating Dependencies

Published runtime dependencies are managed through human-edited ranges in
`requirements.in`; this file is read by `setup.py` into package metadata and
must not contain hash-locked pins. Development, test, lint, and lock-generation
environments are managed through human-edited inputs in `requirements/*.in` and
generated hash-locked files in `requirements/*.txt`.

Edit the relevant `.in` file, regenerate the matching lock using the commands in
`internal/specs/python-dependency-hashes.md`, and commit both files. Do not edit
generated lock files by hand.

See [internal/specs/python-dependency-hashes.md](internal/specs/python-dependency-hashes.md)
for the exact commands.
