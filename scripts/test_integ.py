#!/usr/bin/env python
import subprocess
import sys
subprocess.run(["pytest", "tests-integ"] + sys.argv[1:], check=True)
