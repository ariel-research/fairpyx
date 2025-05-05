"""
Initial failing tests for the (empty) HFFD skeleton.
Run:  pytest -q
"""

import pytest
import numpy as np
import fairpyx
from fairpyx.algorithms.hffd import hffd

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
