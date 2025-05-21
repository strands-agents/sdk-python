#!/usr/bin/env python
import subprocess
import sys

# Run ruff check
subprocess.run(["ruff", "check"], check=True)

# Run mypy
subprocess.run(["mypy", "-p", "src"], check=True)

print("All lint checks passed!")
