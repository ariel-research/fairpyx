import pytest

@pytest.fixture
def x():
    return 5
def test_print(x):
    print(x)