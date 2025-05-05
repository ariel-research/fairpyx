import random, numpy as np
import pytest
from fairpyx.algorithms.hffd import hffd
import fairpyx

# ----------  preserve RNG state so we don't break other tests  ----------
@pytest.fixture(autouse=True)
def _preserve_random_state():
    state_py   = random.getstate()
    state_np   = np.random.get_state()
    yield
    random.setstate(state_py)
    np.random.set_state(state_np)
# -----------------------------------------------------------------------

def test_allocation_not_empty():
    costs = np.array([[8,5,5,2],
                      [8,5,5,2]])
    inst = fairpyx.Instance(valuations=costs, agent_capacities=[10,10])
    allocation = fairpyx.divide(hffd, inst)
    assert any(allocation.values()), "HFFD returned empty allocation"

def test_respect_capacities():
    costs = np.array([[9,8,4],
                      [9,8,4]])
    inst = fairpyx.Instance(valuations=costs, agent_capacities=[10,10])
    allocation = fairpyx.divide(hffd, inst)
    fairpyx.validate_allocation(inst, allocation,
                                title="HFFD capacity check")

