from mip import *
import sys
import logging

from fairpyx import Instance


# TODO - delete this enum from here
class EFTBStatus(Enum):
    NO_EF_TB = 0
    EF_TB = 1
    CONTESTED_EF_TB = 2


# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()


# a = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
# a = {'Alice': {3.5: (1, 1, 0), 3: (1, 0, 1)}
# initial_budgets = {"Alice": 5, "Bob": 4}
def check_envy(instance, student, other_student, a):
    result = []
    # check if student envies in other_student
    for bundle_i in a[student].values():
        for bundle_j in a[other_student].values():
            if instance.agent_bundle_value(student, bundle_j) > instance.agent_bundle_value(student, bundle_i):
                result.append((bundle_i, bundle_j))
    return result


def get_envy_constraints(instance, initial_budgets, a):
    """
        This function checks for every two students if there is envy between them,
        in case there is a constraint required for the model.

        :param instance: a fair-course-allocation instance
        :param initial_budgets:  the initial budgets of the students
        :param a: matrix that says for each budget what is the bundle with the maximum utility that a student can take
        :param x: variables decision of the model

        Example run 6 iteration 5
        >>> instance = Instance(
        ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
        ...     agent_capacities=2,
        ...     item_capacities={"x":1, "y":1, "z":2})
        >>> initial_budgets = {"Alice": 5, "Bob": 4}
        >>> a = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
        >>> get_envy_constraints(instance, initial_budgets, a)
        [(('Alice', ('x', 'z')), ('Bob', ('x', 'y')))]
    """

    students_names = instance.agents
    envy_constraints = []

    for student in students_names:
        for other_student in students_names:
            if student is not other_student:
                if initial_budgets[student] > initial_budgets[other_student]:  # check envy
                    # result contain the index of the bundles that student envious other_student
                    result = check_envy(instance, student, other_student, a)

                    if result:
                        for pair in result:
                            i, j = pair
                            logger.info(f"bundle {i} , bundle {j}")
                            # envy_constraints.append((x[student, i], x[other_student, j]))
                            envy_constraints.append(((student, i), (other_student, j)))
                            logger.info(f"student {student} bundle {i} envy student {other_student} bundle {j}")
    return envy_constraints


def optimize_model(a: dict, instance: Instance, prices: dict, t: Enum, initial_budgets: dict):
    """
        Example run 6 iteration 5
        >>> from fairpyx import Instance
        >>> from fairpyx.algorithms import ACEEI
        >>> instance = Instance(
        ...     valuations={"Alice":{"x":5, "y":4, "z":1}, "Bob":{"x":4, "y":6, "z":3}},
        ...     agent_capacities=2,
        ...     item_capacities={"x":1, "y":1, "z":2})
        >>> a = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
        >>> initial_budgets = {"Alice": 5, "Bob": 4}
        >>> prices = {"x": 1.5, "y": 2, "z": 0}
        >>> t = ACEEI.EFTBStatus.EF_TB
        >>> optimize_model(a,instance,prices,t,initial_budgets)
        [[('x', 'z'), ('x', 'y')], [('y', 'z'), ('x', 'y')]]
    """
    model = Model("allocations")
    courses_names = list(instance.items)
    students_names = list(instance.agents)

    # Decision variables
    x = {(student, bundle): model.add_var(var_type=BINARY) for student in students_names for bundle in a[student].values()}

    z = {course: model.add_var(var_type=CONTINUOUS, lb=-instance.item_capacity(course)) for course in courses_names}
    y = {course: model.add_var(var_type=CONTINUOUS) for course in courses_names}

    # Objective function
    objective_expr = xsum(y[course] for course in courses_names)
    model.objective = minimize(objective_expr)

    # Add constraints for absolute value of excess demand
    for course in courses_names:
        model += y[course] >= z[course]
        model += y[course] >= -z[course]

    # Course allocation constraints
    for course in courses_names:
        # constraint 1: ∑  ∑(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) = 𝑐_𝑗 + 𝑧_𝑗  ∀𝑗 ∈ [𝑚], 𝑝_𝑗 > 0
        #            𝑖∈[𝑛] ℓ ∈ [𝑘_𝑖]
        if prices[course] > 0:
            model += xsum(
                x[student, bundle] * (1 if course in bundle else 0) for student in students_names for bundle in
                a[student].values()) == instance.item_capacity(course) + z[course]

        # constraint 2: ∑     ∑(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) ≤ 𝑐𝑗 + 𝑧𝑗 ∀𝑗 ∈ [𝑚], 𝑝𝑗 = 0
        #  𝑖∈[𝑛] ℓ∈[𝑘_𝑖]
        else:
            model += xsum(
                x[student, bundle] * (1 if course in bundle else 0) for student in students_names for bundle in
                a[student].values()) <= instance.item_capacity(course) + z[course]

    # constraint 3: ∑𝑥_𝑖ℓ = 1  ∀𝑖 ∈ [𝑛]
    #               ℓ∈[𝑘_𝑖]
    for student in students_names:
        model += xsum(x[student, bundle] for bundle in a[student].values()) == 1

    # Add EF-TB constraints based on parameter t
    if t == EFTBStatus.NO_EF_TB:
        pass  # No EF-TB constraints, no need to anything
    elif t == EFTBStatus.EF_TB:
        # Add EF-TB constraints here
        envy_constraints = get_envy_constraints(instance, initial_budgets, a)
        for constraint in envy_constraints:
            model += x[constraint[0]] + x[constraint[1]] <= 1

    elif t == EFTBStatus.CONTESTED_EF_TB:
        # Add contested EF-TB constraints here
        pass

    # Optimize the model
    model.optimize()

    # Process and print results
    if model.num_solutions:
        print("Objective Value:", model.objective_value)
        for student in students_names:
            for l in a[student].values():
                print(f"x_{student}{l} =", x[student, l].x)
        for course in courses_names:
            print(f"|z_{course}|=y_{course} =", y[course].x)
    else:
        print("Optimization was not successful. Status:", model.status)


if __name__ == "__main__":
    instance = Instance(
        valuations={"Alice": {"x": 5, "y": 4, "z": 1}, "Bob": {"x": 4, "y": 6, "z": 3}},
        agent_capacities=2,
        item_capacities={"x": 1, "y": 1, "z": 2}
    )
    a = {'Alice': {3.5: ('x', 'y'), 3: ('x', 'z')}, 'Bob': {3.5: ('x', 'y'), 2: ('y', 'z')}}
    initial_budgets = {"Alice": 5, "Bob": 4}
    prices = {"x": 1.5, "y": 2, "z": 0}
    t = EFTBStatus.EF_TB

    optimize_model(a, instance, prices, t, initial_budgets)
