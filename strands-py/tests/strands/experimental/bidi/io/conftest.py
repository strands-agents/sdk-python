import sys
from types import ModuleType

# The standard test environment does not install the native PyAudio dependency.
sys.modules.setdefault("pyaudio", ModuleType("pyaudio"))
