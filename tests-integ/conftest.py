import pytest
import time

@pytest.fixture(autouse=True)
def sleep_to_avoid_throttling():
    time.sleep(5)
