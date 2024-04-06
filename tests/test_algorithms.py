import pytest
from fairpyx.instances import Instance
import numpy as np
import pytest


def random_uniform_extended(num_of_agents: int, num_of_items: int,
                            num_of_categories: int,
                            agent_capacity_bounds: tuple[int, int],
                            item_capacity_bounds: tuple[int, int],
                            item_base_value_bounds: tuple[int, int],
                            item_subjective_ratio_bounds: tuple[float, float],
                            normalized_sum_of_values: int,
                            agent_name_template="s{index}", item_name_template="c{index}",
                            random_seed: int = None,
                            ):
    inst = Instance(
        valuations={"Agent1": {'item1': 1}})  # helping varialbe (for the sake of calling inner # non-static-methods)
    result_instance = inst.random_uniform(num_of_agents=
                                          num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=
                                          agent_capacity_bounds, item_capacity_bounds=item_capacity_bounds,
                                          item_base_value_bounds=item_base_value_bounds
                                          , item_subjective_ratio_bounds=item_subjective_ratio_bounds
                                          , normalized_sum_of_values=normalized_sum_of_values,
                                          agent_name_template=agent_name_template
                                          , item_name_template=item_name_template, random_seed=random_seed)
    if random_seed is not None:
        random_seed = np.random.randint(1, 2 ** 31)
    np.random.seed(random_seed)

    # TODO make ranmdomize categories
    category_string_template = "Category:{cat}"
    categories = {category_string_template.format(cat=cat): [] for cat in range(num_of_categories)}
    agent_capacities_2d = {
        agent: {category: np.random.randint(agent_capacity_bounds[0], agent_capacity_bounds[1] + 1) for category in
                categories} for agent
        in result_instance.agents}

    # print(f"and categories are :{categories}\n and agent capacities are : {agent_capacities_2d}") # TODO remove after finish
    for item in result_instance.items:
        random_category = np.random.choice(list(categories.keys()))
        categories[random_category].append(item)

    return Instance, agent_capacities_2d, categories


def test_1():
    print(main())
    assert False


def test_2():
    print(main())
    assert False


def test_3():
    print(main())
    assert False


def test_4():
    print(main())
    assert False


def test_5():
    print(main())
    assert False


def test_6():
    print(main())
    assert False


def test_7():
    print(main())
    assert False


def test_8():
    print(main())
    assert False


# @pytest.fixture
def main():  # todo add randomization for arguments .
    random_num_of_agents = np.random.randint(1, 10 + 1)
    random_num_of_items = np.random.randint(1, 10 + 1)
    random_num_of_categories = np.random.randint(1, random_num_of_items + 1)
    random_instance = random_uniform_extended(
        num_of_categories=random_num_of_categories,
        num_of_agents=random_num_of_agents, num_of_items=random_num_of_items,
        agent_capacity_bounds=[1, 20], item_capacity_bounds=[1, 50],
        item_base_value_bounds=[1, 200], item_subjective_ratio_bounds=[0.5, 1.5],
        agent_name_template="agent{index}", item_name_template="item{index}",
        normalized_sum_of_values=1000)
    return random_instance


if __name__ == "__main__":
    main()