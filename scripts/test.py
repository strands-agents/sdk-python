#!/usr/bin/env python
import subprocess
import sys
subprocess.run(["pytest", "--cov", "--cov-config=pyproject.toml"] + sys.argv[1:], check=True)
