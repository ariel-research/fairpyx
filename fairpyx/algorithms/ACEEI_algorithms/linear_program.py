"""
"Practical algorithms and experimentally validated incentives for equilibrium-based fair division (A-CEEI)"
    by ERIC BUDISH, RUIQUAN GAO, ABRAHAM OTHMAN, AVIAD RUBINSTEIN, QIANFAN ZHANG. (2023)
    link to the article: https://arxiv.org/pdf/2305.11406
    Linear programming that used in algorithm 1

Programmers: Erga Bar-Ilan, Ofir Shitrit and Renana Turgeman.
Since: 2024-01
"""
from mip import *
import sys
import logging
from contextlib import redirect_stdout
import os

from fairpyx import Instance
from fairpyx.algorithms import ACEEI_algorithms

logger = logging.getLogger(__name__)


# ---------------------The main function---------------------
def optimize_model(map_student_to_best_bundle_per_budget: dict, instance: Instance, prices: dict, t: Enum, initial_budgets: dict):
    """
        This function calculates a linear program whose goal is to find the distribution
        and budgets that meet the constraints.
        It is used in Algorithm 1.

        :param map_student_to_best_bundle_per_budget: a dictionary that maps each student to his best bundle per budget.
        :param instance: a fair-course-allocation instance
        :param prices: the prices of courses
        :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB
        :param initial_budgets: Students' initial budgets

        :return final courses prices, final budgets, final allocation

        >>> from fairpyx import Instance
        >>> from fairpyx.algorithms import ACEEI_algorithms

        Example run 6 iteration 5
        >>> instance = Instance(
        ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
        ...     agent_capacities=2,
        ...     item_capacities={"x":1, "y":1, "z":2})
        >>> map_student_to_best_bundle_per_budget = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
        >>> initial_budgets = {"Alice": 5, "Bob": 4}
        >>> prices = {"x": 1.5, "y": 2, "z": 0}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB
        >>> optimize_model(map_student_to_best_bundle_per_budget,instance,prices,t,initial_budgets)
        ({'Alice': (3, ('x', 'z')), 'Bob': (2, ('y', 'z'))}, 0.0, {'x': 0.0, 'y': 0.0, 'z': 0.0})

        Example with a student with no bundle
        >>> instance = Instance(
        ...     valuations={"avi":{"x":5}, "beni":{"x":5}},
        ...     agent_capacities=1,
        ...     item_capacities={"x":1})
        >>> map_student_to_best_bundle_per_budget = {'avi': {1.3: ('x',)}, 'beni': {0: ()}}
        >>> initial_budgets = {"avi": 1.1, "beni": 1}
        >>> prices = {"x": 1.3}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB
        >>> optimize_model(map_student_to_best_bundle_per_budget,instance,prices,t,initial_budgets)
        ({'avi': (1.3, ('x',)), 'beni': (0, ())}, 0.0, {'x': 0.0})

    """

    logger.info("\n----START LINEAR_PROGRAM: a = %s, p = %s, t = %s, initial_budgets = %s", map_student_to_best_bundle_per_budget, prices, t, initial_budgets)

    model = Model("allocations")
    courses_names = list(instance.items)
    students_names = list(instance.agents)

    # Decision variables
    x = {(student, bundle): model.add_var(var_type=BINARY) for student in students_names for bundle in
         map_student_to_best_bundle_per_budget[student].values()}

    z = {course: model.add_var(var_type=CONTINUOUS, lb=-instance.item_capacity(course)) for course in courses_names}
    y = {course: model.add_var(var_type=CONTINUOUS) for course in courses_names}

    # Define binary variables δ
    delta = {course: model.add_var(var_type=BINARY) for course in courses_names}
    # Big-M value, should be large enough to cover the range of z
    M = 1e6

    # Objective function
    objective_expr = xsum(y[course] for course in courses_names)
    model.objective = minimize(objective_expr)

    for course in courses_names:
        # Add constraints to define y based on the value of z
        model += y[course] <= z[course] + M * (1 - delta[course])
        model += y[course] >= z[course]
        model += y[course] <= M * delta[course]
        model += z[course] <= M * delta[course]
        model += z[course] >= 0 - M * (1 - delta[course])

    # Course allocation constraints
    for course in courses_names:
        # constraint 1: ∑  ∑(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) = 𝑐_𝑗 + 𝑧_𝑗  ∀𝑗 ∈ [𝑚], 𝑝_𝑗 > 0
        #            𝑖∈[𝑛] ℓ ∈ [𝑘_𝑖]
        if prices[course] > 0:
            model += xsum(
                x[student, bundle] * (1 if course in bundle else 0) for student in students_names for bundle in
                map_student_to_best_bundle_per_budget[student].values()) == instance.item_capacity(course) + z[course]

        # constraint 2: ∑     ∑(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) ≤ 𝑐𝑗 + 𝑧𝑗 ∀𝑗 ∈ [𝑚], 𝑝𝑗 = 0
        #  𝑖∈[𝑛] ℓ∈[𝑘_𝑖]
        else:
            model += xsum(
                x[student, bundle] * (1 if course in bundle else 0) for student in students_names for bundle in
                map_student_to_best_bundle_per_budget[student].values()) <= instance.item_capacity(course) + z[course]

    # constraint 3: ∑𝑥_𝑖ℓ = 1  ∀𝑖 ∈ [𝑛]
    #               ℓ∈[𝑘_𝑖]
    for student in students_names:
        model += xsum(x[student, bundle] for bundle in map_student_to_best_bundle_per_budget[student].values()) == 1

    # Add EF-TB constraints based on parameter t
    if t == ACEEI_algorithms.ACEEI.EFTBStatus.NO_EF_TB:
        pass  # No EF-TB constraints, no need to anything

    elif t == ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB or t == ACEEI_algorithms.ACEEI.EFTBStatus.CONTESTED_EF_TB:
        # Add EF-TB constraints here
        envy_constraints = get_envy_constraints(instance, initial_budgets, map_student_to_best_bundle_per_budget, t, prices)
        for constraint in envy_constraints:
            model += x[constraint[0]] + x[constraint[1]] <= 1

    # Redirect solver output to null device
    model.verbose = 0

    # Optimize the model
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            model.optimize()

    if model.num_solutions:
        excess_demand_per_course = {course: y[course].x for course in courses_names}
        logger.debug("model.optimize found %s solutions. excess_demand_per_course=%s", model.num_solutions, excess_demand_per_course)
    else:
        excess_demand_per_course = model.status
        logger.debug("model.optimize found no solutions.")

    new_budgets = {}
    for (student, bundle), var in x.items():
        if var.x == 1:  # Check if the decision variable is set to 1
            price = list(map_student_to_best_bundle_per_budget[student].keys())[
                list(map_student_to_best_bundle_per_budget[student].values()).index(bundle)]  # Extract the price from dictionary a
            new_budgets[student] = (price, bundle)

    logging.info("\nNew budgets: %s\nObjective Value: %s\nExcess Demand: %s", new_budgets, model.objective_value,
                 excess_demand_per_course)
    logger.info("FINISH LINEAR_PROGRAM\n")

    return new_budgets, model.objective_value, excess_demand_per_course


# ---------------------helper functions:---------------------

def check_envy(instance: Instance, student: str, other_student: str, a: dict, t: Enum, prices: dict):
    """
        The function accepts a pair of students, and returns pairs of courses for which envy exists.

        :param instance: a fair-course-allocation instance
        :param student: The student with the highest initial budget
        :param other_student: The student with the lowest initial budget
        :param a: dict that says for each budget what is the bundle with the maximum utility that a student can take
        :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB
        :param prices: courses prices

        Example run 6 iteration 5
        >>> from fairpyx import Instance
        >>> from fairpyx.algorithms import ACEEI_algorithms

        >>> instance = Instance(
        ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
        ...     agent_capacities=2,
        ...     item_capacities={"x":1, "y":1, "z":2})
        >>> student = "Alice"
        >>> other_student = "Bob"
        >>> a = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB
        >>> prices = {"x": 1.5, "y": 2, "z": 0}
        >>> check_envy(instance, student, other_student, a, t, prices)
        [(('x', 'z'), ('x', 'y'))]

        >>> instance = Instance(
        ...     valuations={"Alice":{"x":10, "y":20}, "Bob":{"x":10, "y":20}},
        ...     agent_capacities=1,
        ...     item_capacities={"x":1, "y":1})
        >>> student = "Alice"
        >>> other_student = "Bob"
        >>> a = {'Alice': {0: (), 1.1: ('y')}, 'Bob': {1.1: ('y'), 1: ('x')}}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB
        >>> prices = {"x": 1, "y": 1.1}
        >>> check_envy(instance, student, other_student, a, t, prices)
        [((), 'y'), ((), 'x')]

        new example
        >>> instance = Instance(
        ...     valuations={"Alice": {"x": 5, "y": 4, "z": 1, "w": 6}, "Bob": {"x": 4, "y": 6, "z": 3, "w": 1}},
        ...     agent_capacities=2,
        ...     item_capacities={"x":1, "y":1, "z":2, "w":1})
        >>> student = "Alice"
        >>> other_student = "Bob"
        >>> a = {'Alice': {3.5: ('x', 'y')}, 'Bob': {3.5: ('x'), 2: ('y', 'z')}}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.CONTESTED_EF_TB
        >>> prices = {"x": 1, "y": 0.1, "z": 0, "w": 0}
        >>> check_envy(instance, student, other_student, a, t, prices)
        [(('x', 'y'), 'x'), (('x', 'y'), ('y', 'z'))]

    """
    result = []
    # check if student envies in other_student
    for bundle_i in a[student].values():
        for bundle_j in a[other_student].values():
            original_bundle_j = bundle_j
            if t == ACEEI_algorithms.ACEEI.EFTBStatus.CONTESTED_EF_TB:
                bundle_j = list(bundle_j)  # Convert bundle_j to a list

                # Iterate through keys in prices
                for key, value in prices.items():
                    # Check if value is 0 and key is not already in bundle_j
                    if value == 0 and key not in bundle_j:
                        # Add key to bundle_j
                        bundle_j.append(key)

                sorted_bundle_j = sorted(bundle_j, key=lambda course: instance.agent_item_value(student, course),
                                         reverse=True)

                sorted_bundle_j = sorted_bundle_j[:instance.agent_capacity(student)]

                bundle_j = tuple(sorted_bundle_j)

            if instance.agent_bundle_value(student, bundle_j) > instance.agent_bundle_value(student, bundle_i):
                result.append((bundle_i, original_bundle_j))

    return result


def get_envy_constraints(instance: Instance, initial_budgets: dict, a: dict, t: Enum, prices: dict):
    """
        This function checks for every two students if there is envy between them,
        in case there is a constraint required for the model.

        :param instance: a fair-course-allocation instance
        :param initial_budgets:  the initial budgets of the students
        :param a: dict that says for each budget what is the bundle with the maximum utility that a student can take
        :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB
        :param prices: courses prices


        Example run 6 iteration 5
        >>> from fairpyx import Instance
        >>> from fairpyx.algorithms import ACEEI_algorithms
        >>> instance = Instance(
        ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
        ...     agent_capacities=2,
        ...     item_capacities={"x":1, "y":1, "z":2})
        >>> initial_budgets = {"Alice": 5, "Bob": 4}
        >>> a = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB
        >>> prices = {"x": 1, "y": 1.1}
        >>> get_envy_constraints(instance, initial_budgets, a, t, prices)
        [(('Alice', ('x', 'z')), ('Bob', ('x', 'y')))]

        >>> instance = Instance(
        ...     valuations={"Alice":{"x":10, "y":20}, "Bob":{"x":10, "y":20}},
        ...     agent_capacities=1,
        ...     item_capacities={"x":1, "y":1})
        >>> initial_budgets = {"Alice": 1.1, "Bob": 1}
        >>> a = {'Alice': {0: (), 1.1: ('y')}, 'Bob': {1.1: ('y'), 1: ('x')}}
        >>> t = ACEEI_algorithms.ACEEI.EFTBStatus.EF_TB
        >>> prices = {"x": 1, "y": 1.1}
        >>> get_envy_constraints(instance, initial_budgets, a, t, prices)
        [(('Alice', ()), ('Bob', 'y')), (('Alice', ()), ('Bob', 'x'))]
    """

    students_names = instance.agents
    envy_constraints = []

    for student in students_names:
        for other_student in students_names:
            if student is not other_student:
                if initial_budgets[student] > initial_budgets[other_student]:  # check envy
                    # result contain the index of the bundles that student envious other_student
                    result = check_envy(instance, student, other_student, a, t, prices)

                    if result:
                        for pair in result:
                            i, j = pair
                            # logger.info(f"bundle {i} , bundle {j}")
                            # envy_constraints.append((x[student, i], x[other_student, j]))
                            envy_constraints.append(((student, i), (other_student, j)))
                            # logger.info(f"student {student} bundle {i} envy student {other_student} bundle {j}")
    return envy_constraints


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())


    # instance = Instance(
    #     valuations={"Alice": {"x": 5, "y": 4, "z": 1, "w": 6}, "Bob": {"x": 4, "y": 6, "z": 3, "w": 1}},
    #     agent_capacities=2,
    #     item_capacities={"x": 1, "y": 1, "z": 2}
    # )
    #
    # a = {'Alice': {3.5: ('x', 'y')}, 'Bob': {3.5: ('x'), 2: ('y', 'z')}}
    # initial_budgets = {"Alice": 1.1, "Bob": 1}
    # prices = {"x": 1, "y": 0.1, "z": 0, "w": 0}
    # t = EFTBStatus.CONTESTED_EF_TB
    #
    # print(check_envy(instance, "Alice", "Bob", a, t, prices))

    # optimize_model(a, instance, prices, t, initial_budgets)
    # result = [(('x', 'y'), ('x')), (('x', 'y'), ('y', 'z'))]

    # instance = Instance(valuations = {"Alice": {"x": 5, "y": 4, "z": 1, "w": 6}, "Bob": {"x": 4, "y": 6, "z": 3, "w": 1}}, agent_capacities = 2, item_capacities = {"x": 1, "y": 1, "z": 2, "w": 1})
    # student = "Alice"
    # other_student = "Bob"
    # a = {'Alice': {3.5: ('x', 'y')}, 'Bob': {3.5: ('x'), 2: ('y', 'z')}}
    # t = EFTBStatus.CONTESTED_EF_TB
    # prices = {"x": 1, "y": 0.1, "z": 0, "w": 0}
    # print(check_envy(instance, student, other_student, a, t, prices))
