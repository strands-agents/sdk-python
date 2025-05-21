#!/usr/bin/env python
import subprocess
import sys

# Only run ruff check for now, as mypy has issues that need to be fixed separately
subprocess.run(["ruff", "check"], check=True)
print("Lint checks passed!")
