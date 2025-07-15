import pytest

## Data


@pytest.fixture
def yellow_img(pytestconfig):
    path = pytestconfig.rootdir / "tests_integ/yellow.png"
    with open(path, "rb") as fp:
        return fp.read()


## Async


@pytest.fixture(scope="session")
def agenerator():
    async def agenerator(items):
        for item in items:
            yield item

    return agenerator


@pytest.fixture(scope="session")
def alist():
    async def alist(items):
        return [item async for item in items]

    return alist
