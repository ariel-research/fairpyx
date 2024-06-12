"""
Test the maximin-aware algorithms

Programmer: Sonya Rybakov
Since:  2024-05
"""

import numpy as np
import pytest

import fairpyx
from fairpyx.algorithms import maximin_aware
from fairpyx.satisfaction import AgentBundleValueMatrix

NUM_OF_RANDOM_INSTANCES = 10


def test_feasibility_for_three():
    """tests both algorithms feasibility for 3 agents, since divide and choose is defined for 3 only"""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        numitems = np.random.randint(1, 100)
        instance = fairpyx.Instance(valuations=np.random.randint(1, 11, (3, numitems)))
        allocation = fairpyx.divide(maximin_aware.divide_and_choose_for_three, instance=instance)
        fairpyx.validate_allocation(instance, allocation,
                                    title=f"mma feasibility: Seed {i} divide and choose")
        allocation = fairpyx.divide(maximin_aware.alloc_by_matching, instance=instance)
        fairpyx.validate_allocation(instance, allocation,
                                    title=f"mma feasibility: Seed {i} allocation by matching for 3")


def test_feasibility_for_any():
    """tests allocation by matching feasibility for any amount of agents"""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        numitems = np.random.randint(1, 100)
        numagents = np.random.randint(2, 100)
        instance = fairpyx.Instance(valuations=np.random.randint(1, 11, (numagents, numitems)))
        allocation = fairpyx.divide(maximin_aware.alloc_by_matching, instance=instance)
        fairpyx.validate_allocation(instance, allocation,
                                    title=f"mma feasibility: Seed {i} allocation by matching for any")


def mma1_fairness_calc(instance: fairpyx.Instance, allocation: dict):
    """
    >>> inst2 = fairpyx.Instance(valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    >>> alloc2 = {'Alice': [1], 'Bob': [0], 'Claire': []}
    >>> mma1_fairness_calc(inst2, alloc2)
    [True, True, True]
    """
    # check if one allocation is lesser by one item
    is_shorter_by_one = [all((len(value) == len(v) - 1) or (len(value) == len(v))
                             for v in allocation.values()) for key, value in allocation.items()]

    abvm = AgentBundleValueMatrix(instance, allocation)
    # normalized matrix gives the percentile worth in comparison to maximum value, i.e. total value
    worthy = [np.round(abvm.normalized_matrix[agent][agent]) >= 33 for agent in abvm.agents]
    return [a or b for a, b in zip(is_shorter_by_one, worthy)]


def test_fairness_divide_and_choose():
    """tests whether the result is MMA1 on randomized inputs
    using the algorithm's proof where each bundle is at least 1/3 worth of all items in all cases"""
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        num_agents = 3
        instance = fairpyx.Instance(valuations=np.random.randint(1, 11, (num_agents, 100)))
        allocation = fairpyx.divide(fairpyx.algorithms.maximin_aware.divide_and_choose_for_three, instance=instance)
        assert any(allocation.values()), f'Seed {i}, mma1: allocation empty'
        assert all(mma1_fairness_calc(instance, allocation)), f'Seed {i}, mma1: fairness failed'


def mma_fairness_calc(instance: fairpyx.Instance, allocation: dict):
    """
    calculates the mma fairness of allocation by matching (algorithm 2)

    :param instance: the allocation problem instance
    :param allocation: the resulted allocation to be checked
    :return: a list where list[i] says whether agent i is mma satisfied

    the idea is based on the algorithm's proof:
    *   mms_i(A-i, n-1) i.e. maximin-share value for reallocation of the remainder abjects for agent i is
        at most the average value of bundles allocated to agents in N1 ∪ N3
        # N1: agents not envied by i
        # N3: agents envied by i with >1 items
    *   mms_i(A-i, n-1) <= 2*v_i(A_i)

    example:
    >>> inst = fairpyx.Instance(valuations={"Alice": [11, 22], "Bob": [33, 44]})
    >>> alloc = {"Alice": [0], "Bob": [1]}
    >>> mma_fairness_calc(inst, alloc)
    [True, True]
    """
    abvm = AgentBundleValueMatrix(instance, allocation, normalized=False)
    abvm.make_envy_matrix()
    mat = abvm.envy_matrix  # mat example {'Alice': {'Alice': 0, 'Bob': 11}, 'Bob': {'Alice': -11, 'Bob': 0}}
    total_satisfaction = []
    for agent in instance.agents:
        unified_bundle = []
        count = 0
        agent_status = mat[agent]
        for k, v in agent_status.items():
            if k != agent and (v <= 0 or (v > 0 and len(allocation[k]) > 1)):
                unified_bundle.extend(allocation[k])
                count += 1

        average_value = instance.agent_bundle_value(agent, unified_bundle) / count if unified_bundle else 0
        # mms_i(A-i, n-1) <= 2*v_i(A_i)
        no_envy = average_value <= 2 * instance.agent_bundle_value(agent, allocation[agent])
        total_satisfaction.append(no_envy)
    return total_satisfaction


def test_fairness_alloc_by_matching():
    """
    tests 1/2mma fairness of alloc_by_matching on randomized inputs
    """
    for i in range(NUM_OF_RANDOM_INSTANCES):
        np.random.seed(i)
        num_agents = np.random.randint(2, 11)
        instance = fairpyx.Instance(valuations=np.random.randint(1, 11, (num_agents, 100)))
        allocation = fairpyx.divide(maximin_aware.alloc_by_matching, instance=instance)
        assert any(allocation.values()), f'Seed {i}, mma by matching: allocation empty'
        total_satisfaction = mma_fairness_calc(instance, allocation)
        assert len(total_satisfaction) == num_agents, f'Seed {i}, mma by matching: missing agents in total'
        assert all(total_satisfaction), f'Seed {i}, mma by matching: fairness failed'


def test_errors_divide_and_choose():
    with pytest.raises(ValueError):
        instance = fairpyx.Instance(valuations={"Alice": [11, 22], "Bob": [33, 44]})
        allocation = fairpyx.divide(maximin_aware.divide_and_choose_for_three, instance=instance)

    with pytest.raises(ValueError, match='divide and choose: item capacity restricted to only one'):
        instance = fairpyx.Instance(agent_capacities={"Alice": 3, "Bob": 3, "Claire": 3},
                                    item_capacities={"c1": 4, "c2": 5, "c3": 4},
                                    valuations={"Alice": {"c1": 11, "c2": 22, "c3": 11},
                                                "Bob": {"c1": 33, "c2": 44, "c3": 11},
                                                "Claire": {"c1": 33, "c2": 44, "c3": 11}})
        allocation = fairpyx.divide(fairpyx.algorithms.maximin_aware.divide_and_choose_for_three, instance=instance)

    with pytest.raises(ValueError, match='divide and choose: agent capacity should be as many as the items'):
        instance = fairpyx.Instance(agent_capacities={"Alice": 3, "Bob": 2, "Claire": 3},
                                    item_capacities={"c1": 1, "c2": 1, "c3": 1},
                                    valuations={"Alice": {"c1": 11, "c2": 22, "c3": 11},
                                                "Bob": {"c1": 33, "c2": 44, "c3": 11},
                                                "Claire": {"c1": 33, "c2": 44, "c3": 11}})
        allocation = fairpyx.divide(fairpyx.algorithms.maximin_aware.divide_and_choose_for_three, instance=instance)


def test_errors_alloc_by_matching():
    with pytest.raises(ValueError, match='alloc by matching: item capacity restricted to only one'):
        instance = fairpyx.Instance(agent_capacities={"Alice": 3, "Bob": 3, "Claire": 3},
                                    item_capacities={"c1": 4, "c2": 5, "c3": 4},
                                    valuations={"Alice": {"c1": 11, "c2": 22, "c3": 11},
                                                "Bob": {"c1": 33, "c2": 44, "c3": 11},
                                                "Claire": {"c1": 33, "c2": 44, "c3": 11}})
        allocation = fairpyx.divide(fairpyx.algorithms.maximin_aware.alloc_by_matching, instance=instance)

    with pytest.raises(ValueError, match='alloc by matching: agent capacity should be as many as the items'):
        instance = fairpyx.Instance(agent_capacities={"Alice": 3, "Bob": 2, "Claire": 3},
                                    item_capacities={"c1": 1, "c2": 1, "c3": 1},
                                    valuations={"Alice": {"c1": 11, "c2": 22, "c3": 11},
                                                "Bob": {"c1": 33, "c2": 44, "c3": 11},
                                                "Claire": {"c1": 33, "c2": 44, "c3": 11}})
        allocation = fairpyx.divide(fairpyx.algorithms.maximin_aware.alloc_by_matching, instance=instance)


def test_divide_and_choose():
    inst = fairpyx.Instance(valuations={"Alice": [9, 10], "Bob": [7, 5], "Claire": [2, 8]})
    alloc = fairpyx.divide(maximin_aware.divide_and_choose_for_three, inst)
    assert alloc == {'Alice': [1], 'Bob': [0], 'Claire': []}, f'mma1: step 2 allocation incorrect'
    assert all(mma1_fairness_calc(inst, alloc)), f'mma1: step 2 fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [10, 10, 6, 4, 2, 2, 2], "Bob": [7, 5, 6, 6, 6, 2, 9], "Claire": [2, 9, 8, 7, 5, 2, 3]})
    alloc = fairpyx.divide(maximin_aware.divide_and_choose_for_three, inst)
    assert alloc == {'Alice': [0, 1, 5], 'Bob': [2, 6], 'Claire': [3, 4]}, f'mma1: step 2 allocation incorrect'
    assert all(mma1_fairness_calc(inst, alloc)), f'mma1: step 2 fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [10, 10, 6, 4], "Bob": [7, 5, 6, 6], "Claire": [2, 8, 8, 7]})
    alloc = fairpyx.divide(maximin_aware.divide_and_choose_for_three, inst)
    assert alloc == {'Alice': [0], 'Bob': [2, 3], 'Claire': [1]}, f'mma1: step 3 allocation incorrect'
    assert all(mma1_fairness_calc(inst, alloc)), f'mma1: step 3 fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [2,2,6,7], "Bob": [5,7,3,5], "Claire": [2, 2, 2, 2]})
    alloc = fairpyx.divide(maximin_aware.divide_and_choose_for_three, inst)
    assert alloc == {'Alice': [2], 'Bob': [1], 'Claire': [0, 3]}, f'mma1: step 4-I allocation incorrect'
    assert all(mma1_fairness_calc(inst, alloc)), f'mma1: step 4-I fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [2,4,6,7], "Bob": [5,7,3,5], "Claire": [2, 2, 2, 2]})
    alloc = fairpyx.divide(maximin_aware.divide_and_choose_for_three, inst)
    assert alloc == {'Alice': [3], 'Bob': [0, 2], 'Claire': [1]}, f'mma1: step 4-II allocation incorrect'
    assert all(mma1_fairness_calc(inst, alloc)), f'mma1: step 4-II fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [8, 5, 1, 5, 5, 3, 6, 9, 3, 3, 7, 5, 8, 8, 4, 10, 3, 8, 10, 2],
                    "Bob": [3, 5, 5, 3, 4, 9, 5, 5, 8, 1, 2, 6, 8, 6, 9, 1, 2, 8, 9, 7],
                    "Claire": [7, 1, 2, 9, 3, 2, 3, 8, 8, 7, 4, 10, 10, 6, 9, 10, 5, 3, 10, 3]})
    alloc = fairpyx.divide(maximin_aware.divide_and_choose_for_three, inst)
    assert alloc == {'Alice': [0, 1, 4, 6, 12, 13, 16, 19], 'Bob':[2, 5, 8, 9, 15, 17, 18],
                     'Claire':  [3, 7, 10, 11, 14]}, f'mma1: large input allocation incorrect'
    assert all(mma1_fairness_calc(inst, alloc)), f'mma1: large fairness failed'


def test_alloc_by_matching():
    inst = fairpyx.Instance(valuations={"Alice": [10, 10, 6, 4], "Bob": [7, 5, 6, 6], "Claire": [2, 8, 8, 7]})
    alloc = fairpyx.divide(maximin_aware.alloc_by_matching, inst)
    assert alloc == {'Alice': [1], 'Bob': [0], 'Claire': [2, 3]}, f'mma by matching: allocation incorrect'
    assert all(mma_fairness_calc(inst, alloc)), f'mma by matching: fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [10, 10, 6, 4, 2, 2, 2], "Bob": [7, 5, 6, 6, 6, 2, 9], "Claire": [2, 8, 8, 7, 5, 2, 3]})
    alloc = fairpyx.divide(maximin_aware.alloc_by_matching, inst)
    assert alloc == {'Alice': [0, 2, 5], 'Bob': [4, 6], 'Claire': [1, 3]}, f'mma by matching: allocation incorrect'
    assert all(mma_fairness_calc(inst, alloc)), f'mma by matching: fairness failed'

    inst = fairpyx.Instance(
        valuations={"Alice": [8, 5, 1, 5, 5, 3, 6, 9, 3, 3, 7, 5, 8, 8, 4, 10, 3, 8, 10, 2],
                    "Bob": [3, 5, 5, 3, 4, 9, 5, 5, 8, 1, 2, 6, 8, 6, 9, 1, 2, 8, 9, 7],
                    "Claire": [7, 1, 2, 9, 3, 2, 3, 8, 8, 7, 4, 10, 10, 6, 9, 10, 5, 3, 10, 3]})
    alloc = fairpyx.divide(maximin_aware.alloc_by_matching, inst)
    assert alloc == {'Alice': [0, 4, 6, 7, 10, 15, 18], 'Bob': [1, 2, 5, 8, 14, 17, 19],
                     'Claire': [3, 9, 11, 12, 13, 16]}, f'mma by matching: large input allocation incorrect'
    assert all(mma_fairness_calc(inst, alloc)), f'mma by matching: large input fairness failed'


if __name__ == "__main__":
    pytest.main(["-v", __file__])
