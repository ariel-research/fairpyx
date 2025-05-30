"""
Test the maximally proportional allocation algorithm.

Programmer: Elroi Carmel
Since:  2025-05
"""

import pytest
from fairpyx import Instance, divide, validate_allocation, AgentBundleValueMatrix
from fairpyx.algorithms import maximally_proportional_allocation


def test_feasibility():
    ntest = 10
    for i in range(ntest):
        instance = Instance.random_uniform(
            num_of_agents=4,
            num_of_items=6,
            item_capacity_bounds=(1, 1),
            agent_capacity_bounds=(6, 6),
            item_base_value_bounds=(60, 100),
            item_subjective_ratio_bounds=(0.6, 1.4),
            normalized_sum_of_values=100,
            random_seed=i,
        )
        allocation = divide(maximally_proportional_allocation, instance)
        validate_allocation(allocation=allocation, instance=instance)


def test_proportional():
    """
    All of the agents should accept proportional bundles
    """

    def is_proportional(allocation: dict, instance: Instance):
        all_items = list(instance.items)
        for agent, bundle in allocation.items():
            agent_propor_share = (
                instance.agent_bundle_value(agent, all_items) / instance.num_of_agents
            )
            if instance.agent_bundle_value(agent, bundle) < agent_propor_share:
                return False
        return True

    instance = Instance(valuations=[[21, 49, 12, 81], [89, 63, 22, 75]])
    alloc = divide(maximally_proportional_allocation, instance)
    assert is_proportional(alloc, instance) == True

    # random instance
    RAND_SEED = 23
    instance = Instance.random_uniform(
        num_of_agents=5,
        num_of_items=10,
        item_capacity_bounds=(1, 1),
        agent_capacity_bounds=(10, 10),
        item_base_value_bounds=(20, 25),
        item_subjective_ratio_bounds=(0.5, 3.0),
        normalized_sum_of_values=50,
        random_seed=RAND_SEED,
    )
    alloc = divide(maximally_proportional_allocation, instance)
    assert (
        is_proportional(alloc, instance) == True
    ), "At least one player did not get a proportional bundle"


def test_maximallity():
    """
    When there are no complete proportional allocation available
    the algorithm maximizes the amount of players which received
    minimal bundle
    """

    def num_agents_received(alloc: dict) -> int:
        return sum(1 for agent in alloc if len(alloc[agent]))

    # Example 2 from paper
    instance = Instance(
        valuations=[
            [45, 35, 10, 10],
            [35, 45, 10, 10],
            [45, 10, 35, 10],
            [66, 4, 10, 20],
        ]
    )
    alloc = divide(maximally_proportional_allocation, instance)
    assert num_agents_received(alloc) == 3

    # Example 3 from paper
    instance = Instance(
        valuations=[
            [78, 12, 10, 2],
            [55, 30, 10, 5],
            [35, 30, 25, 10],
            [48, 22, 20, 10],
        ]
    )
    alloc = divide(maximally_proportional_allocation, instance)
    assert num_agents_received(alloc) == 3

    # Example 4 from paper

    instance = Instance(
        valuations=[
            [20, 20, 31, 25, 2, 2],
            [31, 25, 20, 20, 2, 2],
            [5, 7, 3, 4, 80, 1],
            [4, 4, 4, 4, 4, 80],
        ]
    )
    alloc = divide(maximally_proportional_allocation, instance)
    assert num_agents_received(alloc) == 4
    # in this example also all items are given
    items_given = sum(alloc.values(), [])
    items_given.sort()
    assert items_given == sorted(instance.items)


def test_maxminimallity():
    """
    When there are multiple maxmiall allocations the alogrithm
    should return allocation that maximizes the minimum rank, over all
    players who receive a minimal bundle, of the most preferred minimal bundle
    """

    # test found by experimenting random instances
    valuation = [
        [193, 127, 413, 53, 89, 124],
        [109, 111, 384, 99, 136, 160],
        [96, 213, 239, 151, 73, 227],
        [176, 190, 212, 128, 97, 197],
    ]
    instance = Instance(valuations=valuation)

    alloc = divide(maximally_proportional_allocation, instance)
    assert alloc == {0: [2], 1: [0, 3, 4], 2: [1, 5], 3: []}


def test_pareto_optimal():
    """
    If there are multiple complete/maximall allocations
    the alogrithm should choose the pareto optimal one
    """
    instance = Instance(valuations=[[35, 40, 25], [40, 35, 25], [40, 25, 35]])
    alloc = divide(maximally_proportional_allocation, instance)
    assert alloc == {0: [1], 1: [0], 2: [2]}


def test_envy_free():
    """
    With 2 agents, a complete proportional allocation is also envy free
    """
    instance = Instance(valuations=[[35, 30, 10, 25], [25, 21, 34, 20]])
    alloc = divide(maximally_proportional_allocation, instance)
    matrix = AgentBundleValueMatrix(instance, alloc, normalized=False)
    matrix.make_envy_matrix()
    # any agent should not get empty bundle
    assert len(next(iter(alloc.values()))) > 0
    assert matrix.max_envy() <= 0


def test_large_input():
    nagents, nitems = 20, 30
    title = f"Large Input. Agents - {nagents}. Items - {nitems}"
    instance = Instance.random_uniform(
        num_of_agents=nagents,
        num_of_items=nitems,
        item_capacity_bounds=(1, 1),
        agent_capacity_bounds=(nitems, nitems),
        item_base_value_bounds=(20, 100),
        item_subjective_ratio_bounds=(0.5, 2.4),
        normalized_sum_of_values=1000,
        random_seed=445,
    )
    alloc = divide(maximally_proportional_allocation, instance)
    validate_allocation(instance=instance, allocation=alloc, title=title)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
