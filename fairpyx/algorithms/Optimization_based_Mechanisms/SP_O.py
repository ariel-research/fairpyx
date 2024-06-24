"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""
import cvxpy
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import cvxpy as cp
import logging
import fairpyx.algorithms.Optimization_based_Mechanisms.optimal_functions as optimal
import fairpyx.algorithms.Optimization_based_Mechanisms.TTC_O as TTC_O
logger = logging.getLogger(__name__)

# global dict
map_student_to_his_sum_bids = {}

def effective_value_with_price(alloc, student, course):
    global map_student_to_his_sum_bids
    return alloc.effective_value(student, course) + map_student_to_his_sum_bids[student]

# conditions (14) in the article using only for the SP-O
def conditions_14(alloc, v,p):
    constraints = []
    for j, course in enumerate(alloc.remaining_items()):
        constraints.append(p[j] >= 0)
    for i, student in enumerate(alloc.remaining_agents()):
        constraints.append(v[i] >= 0)
    return constraints


def SP_O_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorithm 4: Allocate the given items to the given agents using the SP-O protocol.

    SP-O in each round distributes one course to each student, with the refund of the bids according to the price of
    the course. Uses linear planning for optimality.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).


    #>>> from fairpyx.adaptors import divide
    #>>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    #>>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    #>>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    #>>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    #>>> valuations = {"s1": s1, "s2": s2}
    #>>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    #>>> divide(SP_O_function, instance=instance)
    {'s1': ['c2'], 's2': ['c1']}

    #>>> s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}   #{c1: 40, c4: 30, c2:20, c3: 10}
    #>>> s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}     #{c3: 70, c2: 20, c1:6, c4: 4}
    #>>> s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}    #{c4: 50, c3: 21, c2:20, c1: 9}
    #>>> s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}    #{c4: 55, c1: 25, c3:15, c2: 5}
    #>>> s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}      #{c2: 90, c1: 5, c3:3, c4: 2}
    #>>> agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2}
    #>>> item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2}
    #>>> valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    #>>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    #>>> divide(SP_O_function, instance=instance)
    {'s1': ['c1', 'c2'], 's2': ['c1', 'c3'], 's3': ['c3', 'c4'], 's4': ['c1', 'c4'], 's5': ['c2']}
    """

    explanation_logger.info("\nAlgorithm SP-O starts.\n")
    global map_student_to_his_sum_bids

    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in
                         alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    logger.info("Max iterations: %d", max_iterations)

    # the amount of bids agent have from all the courses he got before
    map_student_to_his_sum_bids = {s: 0 for s in alloc.remaining_agents()}

    for iteration in range(max_iterations):
        logger.info("\nIteration number: %d", iteration+1)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            logger.info("There are no more agents (%d) or items(%d) ", len(alloc.remaining_agent_capacities),len(alloc.remaining_item_capacities))
            break

        logger.info("map_student_to_his_sum_bids : " + str(map_student_to_his_sum_bids))
        result_Zt1, result_Zt2, x, problem, rank_mat = TTC_O.roundTTC_O(alloc, logger, effective_value_with_price, 1)  # This is the TTC-O round.

        optimal_value = problem.value
        logger.info("Optimal Objective Value: %d", optimal_value)

        # SP-O
        # condition number 12:
        D = cvxpy.Variable()
        p = cvxpy.Variable(len(alloc.remaining_items()))  # list of price to each course
        v = cvxpy.Variable(len(alloc.remaining_agents()))  # list of value to each student

        # conditions (12) (16) in the article using only for the SP-O
        linear_problem = (result_Zt1 * D + cp.sum([alloc.remaining_item_capacities[course] *
                                                  p[j] for j, course in enumerate(alloc.remaining_items())]) +
                                                    cp.sum([v[i] for i, student in enumerate(alloc.remaining_agents())]))
        # linear problem number 12:
        objective_Wt1 = cp.Minimize(linear_problem)

        constraints_Wt1 = []

        # condition number 13 + 14:
        for j, course in enumerate(alloc.remaining_items()):
            for i, student in enumerate(alloc.remaining_agents()):
                if (alloc.effective_value(student,course) < 0):
                    constraints_Wt1.append(p[j] + v[i] + rank_mat[j][i] * D >= 0)
                else:
                    constraints_Wt1.append(p[j]+v[i]+rank_mat[j][i]*D >= alloc.effective_value(student,course))

        condition_14 = conditions_14(alloc, v, p)
        constraints_Wt1 += condition_14

        problem = cp.Problem(objective_Wt1, constraints=constraints_Wt1)
        result_Wt1 = problem.solve()  # This is the optimal value of program (12)(13)(14).

        logger.info("result_Wt1 - the optimum Wt1: %s", result_Wt1)

        # linear problem number 15:
        objective_Wt2 = cp.Minimize(cp.sum([alloc.remaining_item_capacities[course] * p[j]
                                           for j, course in enumerate(alloc.remaining_items())]))

        constraints_Wt2 = []

        constraints_Wt2.append(linear_problem == result_Wt1)
        constraints_Wt2 += condition_14

        problem = cp.Problem(objective_Wt2, constraints=constraints_Wt2)
        result_Wt2 = problem.solve()  # This is the optimal price

        logger.info("result_Wt2 - the optimum Wt2: %s", result_Wt2)

        # Check if the optimization problem was successfully solved
        if result_Wt2 is not None:
            x_values = x.value
            p_values = p.value
            v_value = v.value
            D_value = D.value
            logger.info("p values: " + str(p_values))
            logger.info("v values: " + str(v_value))
            logger.info("D values: " + str(D_value))

            # Iterate over students and courses to pay the price of the course each student got
            for i, student in enumerate(alloc.remaining_agents()):
                for j, course in enumerate(alloc.remaining_items()):
                    if x_values[j, i] == 1:
                        map_student_to_his_sum_bids[student] -= p_values[j]
                        logger.info("student %s payed: %f", student, p_values[j])
                        logger.info("student %s have in dict: %f", student, map_student_to_his_sum_bids[student])
            optimal.give_items_according_to_allocation_matrix(alloc, x, logger)

            optimal_value = problem.value
            logger.info("Optimal Objective Value: %d", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")


if __name__ == "__main__":
    import doctest, sys, numpy as np
    print("\n", doctest.testmod(), "\n")
    from fairpyx.adaptors import divide

    s1 = {"c1": 10}
    s2 = {"c1": 11}
    instance = Instance(
        agent_capacities={"s1": 1, "s2": 1},
        item_capacities={"c1": 1},
        valuations={"s1": s1, "s2": s2}
    )
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    allocation = divide(SP_O_function, instance=instance)
    print(allocation,"\n\n\n")


    s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}
    s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}
    s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}
    s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}
    s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}
    instance = Instance(
        agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2},
        item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2},
        valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    )
    allocation = divide(SP_O_function, instance=instance)
    print(allocation)

