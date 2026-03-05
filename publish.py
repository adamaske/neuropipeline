#!/usr/bin/env python3
"""Builds and publishes neuropipeline to PyPI."""

import subprocess
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
PACKAGE_DIR = ROOT / "neuropipeline"
TOKEN_FILE = PACKAGE_DIR / "pypi_token.txt"
DIST_DIR = PACKAGE_DIR / "dist"


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    # Read token
    if not TOKEN_FILE.exists():
        print(f"ERROR: Token file not found at {TOKEN_FILE}")
        sys.exit(1)
    token = TOKEN_FILE.read_text().strip()
    if not token:
        print("ERROR: Token file is empty.")
        sys.exit(1)

    # Ensure build tools
    run([sys.executable, "-m", "pip", "install", "-q", "build", "twine"])

    # Clean old dist
    print("\nCleaning old dist...")
    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    for ei in PACKAGE_DIR.glob("*.egg-info"):
        shutil.rmtree(ei)

    # Build
    print("\nBuilding package...")
    run([sys.executable, "-m", "build", str(PACKAGE_DIR)])

    # Upload
    print("\nUploading to PyPI...")
    dist_files = list(DIST_DIR.glob("*"))
    if not dist_files:
        print("ERROR: No dist files found after build.")
        sys.exit(1)
    run([
        sys.executable, "-m", "twine", "upload",
        *[str(f) for f in dist_files],
        "-u", "__token__",
        "-p", token,
    ])

    print("\nDone! Package published successfully.")


if __name__ == "__main__":
    main()
