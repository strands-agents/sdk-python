import concurrent

import pytest
import strands


@pytest.fixture
def thread_pool():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def thread_pool_wrapper(thread_pool):
    return strands.tools.ThreadPoolExecutorWrapper(thread_pool)


def test_submit(thread_pool_wrapper):
    def fun(a, b):
        return (a, b)

    future = thread_pool_wrapper.submit(fun, 1, b=2)

    tru_result = future.result()
    exp_result = (1, 2)

    assert tru_result == exp_result


def test_as_completed(thread_pool_wrapper):
    def fun(i):
        return i

    futures = [thread_pool_wrapper.submit(fun, i) for i in range(2)]

    tru_results = sorted(future.result() for future in thread_pool_wrapper.as_completed(futures))
    exp_results = [0, 1]

    assert tru_results == exp_results


def test_shutdown(thread_pool_wrapper):
    thread_pool_wrapper.shutdown()

    with pytest.raises(RuntimeError):
        thread_pool_wrapper.submit(lambda: None)
