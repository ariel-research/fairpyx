import pytest
from fairpyx import Instance, divide
from fairpyx.algorithms import ACEEI


# Each student will get all the courses
def test_case1():
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=500, item_capacity_bounds=[200, 200])
    allocation = divide(ACEEI, instance=instance)
    for agent in range(instance.num_of_agents):
        for item in range(instance.num_of_items):
            assert (item in allocation[agent])


# Each student i will get course i
def test_case2():
    utilities = [{"{}".format(i): [1 if j == i - 1 else 0 for j in range(100)]} for i in range(1, 101)]
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=100, item_capacity_bounds=[1, 1],
                                       item_base_value_bounds=utilities)
    allocation = divide(ACEEI, instance=instance)
    for i in range(instance.num_of_agents):
        assert (i in allocation[i])


# Each student i will get course i, because student i have the highest i budget.
def test_case3():
    utilities = [{"{}".format(i): list(range(100, i-1, -1))} for i in range(1, 101)]
    instance = Instance.random_uniform(num_of_agents=100, num_of_items=100, item_capacity_bounds=[1, 1],
                                       item_base_value_bounds=utilities)
    b0 = list(range(100, 0, -1))
    allocation = divide(ACEEI, instance=instance, *b0)
    for i in range(instance.num_of_agents):
        assert (i in allocation[i])


if __name__ == "__main__":
    pytest.main(["-v", __file__])
