"""
validityBase Python Utilities
"""

from pathlib import Path

from setuptools import find_packages, setup

ROOT_DIR = Path(__file__).parent


def read_requirements(path):
    """Read non-comment requirement lines from a human-edited input file."""
    return [
        line.strip()
        for line in (ROOT_DIR / path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


long_description = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
requirements = read_requirements("requirements.in")

setup(
    name="vbase-utils",
    version="0.0.1",
    author="validityBase",
    author_email="tech@vbase.com",
    description="validityBase Python Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/validityBase/vbase-py-utils",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
