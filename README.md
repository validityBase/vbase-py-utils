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
    python -m pip install --require-hashes -r requirements-dev.txt
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

## Updating Dependencies

Runtime and development dependencies are managed through human-edited `.in`
files and generated hash-locked `.txt` files. Edit the relevant `.in` file,
regenerate the matching lock with `pip-compile --generate-hashes`, and commit
both files. Do not edit generated lock files by hand.

See [internal/specs/python-dependency-hashes.md](internal/specs/python-dependency-hashes.md)
for the exact commands.
