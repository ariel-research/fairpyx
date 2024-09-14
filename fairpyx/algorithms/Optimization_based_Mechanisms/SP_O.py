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


def SP_O_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger(),  solver=None):
    """
    Algorithm 4: Allocate the given items to the given agents using the SP-O protocol.

    SP-O in each round distributes one course to each student, with the refund of the bids according to the price of
    the course. Uses linear planning for optimality.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).

    :param solver: solver for cvxpy. Default is depend on the version.

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    >>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    >>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    >>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(SP_O_function, instance=instance)
    {'s1': ['c2'], 's2': ['c1']}

    >>> s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}   #{c1: 40, c4: 30, c2:20, c3: 10}
    >>> s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}     #{c3: 70, c2: 20, c1:6, c4: 4}
    >>> s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}    #{c4: 50, c3: 21, c2:20, c1: 9}
    >>> s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}    #{c4: 55, c1: 25, c3:15, c2: 5}
    >>> s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}      #{c2: 90, c1: 5, c3:3, c4: 2}
    >>> agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2}
    >>> item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2}
    >>> valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    >>> divide(SP_O_function, instance=instance)
    {'s1': ['c1'], 's2': ['c2', 'c3'], 's3': ['c3', 'c4'], 's4': ['c1', 'c4'], 's5': ['c1', 'c2']}
    """

    explanation_logger.info("\nAlgorithm SP-O starts.\n")
    global map_student_to_his_sum_bids
    all_agents = set(alloc.remaining_agents())
    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in
                         alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    explanation_logger.debug("Max iterations: %d", max_iterations)

    # the amount of bids agent have from all the courses he got before
    map_student_to_his_sum_bids = {s: 0 for s in alloc.remaining_agents()}

    rank_mat = optimal.createRankMat(alloc, explanation_logger)
    for iteration in range(max_iterations):
        explanation_logger.info("\nIteration number: %d", iteration+1)
        agents_who_need_an_item_in_current_iteration = set(
            alloc.remaining_agents())  # only the agents that still need courses

        # Find the difference between all_agents and agents_who_need_an_item_in_current_iteration
        agents_not_in_need = all_agents - agents_who_need_an_item_in_current_iteration
        # If you need the result as a list:
        agents_not_in_need_list = list(agents_not_in_need)
        for student in agents_not_in_need_list:
            explanation_logger.info("There are no more items you can get", agents=student)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            explanation_logger.info("There are no more agents (%d) or items(%d) ", len(alloc.remaining_agent_capacities),len(alloc.remaining_item_capacities))
            break

        explanation_logger.debug("map_student_to_his_sum_bids : " + str(map_student_to_his_sum_bids))

        result_Zt1, result_Zt2, x, problem = TTC_O.roundTTC_O(alloc, logger, effective_value_with_price, 1, rank_mat, solver)  # This is the TTC-O round.

        explanation_logger.debug("Rank matrix:\n%s", rank_mat)
        optimal_value = problem.value
        explanation_logger.debug("Optimal Objective Value: %d", optimal_value)

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
        explanation_logger.debug("solver: %s", solver)
        result_Wt1 = problem.solve(solver=solver)  # This is the optimal value of program (12)(13)(14).

        explanation_logger.debug("result_Wt1 - the optimum Wt1: %s", result_Wt1)

        # linear problem number 15:
        objective_Wt2 = cp.Minimize(cp.sum([alloc.remaining_item_capacities[course] * p[j]
                                           for j, course in enumerate(alloc.remaining_items())]))

        constraints_Wt2 = []

        constraints_Wt2.append(linear_problem == result_Wt1)
        constraints_Wt2 += condition_14

        problem = cp.Problem(objective_Wt2, constraints=constraints_Wt2)
        result_Wt2 = problem.solve(solver=solver)  # This is the optimal price

        explanation_logger.debug("result_Wt2 - the optimum Wt2: %s", result_Wt2)

        # Check if the optimization problem was successfully solved
        if result_Wt2 is not None:
            x_values = x.value
            p_values = p.value
            v_value = v.value
            D_value = D.value
            explanation_logger.debug("p values: " + str(p_values))
            explanation_logger.debug("v values: " + str(v_value))
            explanation_logger.debug("D values: " + str(D_value))

            # Iterate over students and courses to pay the price of the course each student got
            for i, student in enumerate(alloc.remaining_agents()):
                for j, course in enumerate(alloc.remaining_items()):
                    if x_values[j, i] == 1:
                        #remove the price of the course and save the bids to pass on
                        map_student_to_his_sum_bids[student] -= p_values[j]
                        map_student_to_his_sum_bids[student] += alloc.effective_value(student, course)
                        explanation_logger.info("student %s payed: %f for course %s", student, p_values[j], course, agents=student)
                        explanation_logger.debug("student %s have in dict: %f", student, map_student_to_his_sum_bids[student])
            rank_mat = optimal.give_items_according_to_allocation_matrix(alloc, x, explanation_logger, rank_mat)

            optimal_value = problem.value
            explanation_logger.debug("Optimal Objective Value: %d", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            explanation_logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")


if __name__ == "__main__":
    import doctest, sys, numpy as np
    print("\n", doctest.testmod(), "\n")
    # from fairpyx.adaptors import divide
    #
    # s1 = {"c1": 10}
    # s2 = {"c1": 11}
    # instance = Instance(
    #     agent_capacities={"s1": 1, "s2": 1},
    #     item_capacities={"c1": 1},
    #     valuations={"s1": s1, "s2": s2}
    # )
    #
    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.INFO)
    # # allocation = divide(SP_O_function, instance=instance)
    # # print(allocation,"\n\n\n")
    #
    #
    # s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}
    # s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}
    # s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}
    # s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}
    # s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}
    # instance = Instance(
    #     agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2},
    #     item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2},
    #     valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    # )
    # # allocation = divide(SP_O_function, instance=instance)
    #
    #
    # s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}  # {c1: 40, c4: 30, c2:20, c3: 10}
    # s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}  # {c3: 70, c2: 20, c1:6, c4: 4}
    # s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}  # {c4: 50, c3: 21, c2:20, c1: 9}
    # s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}  # {c4: 55, c1: 25, c3:15, c2: 5}
    # s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}  # {c2: 90, c1: 5, c3:3, c4: 2}
    # agent_capacities = {"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2}
    # item_capacities = {"c1": 3, "c2": 2, "c3": 2, "c4": 2}
    # valuations = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    # instance = Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
    # solver = None
    # allocation = divide(SP_O_function, instance=instance, solver=solver)
    # print(allocation)

    from fairpyx.adaptors import divide_random_instance, divide
    from fairpyx.explanations import ConsoleExplanationLogger, FilesExplanationLogger, StringsExplanationLogger

    num_of_agents = 5
    num_of_items = 3

    # console_explanation_logger = ConsoleExplanationLogger(level=logging.INFO)
    # files_explanation_logger = FilesExplanationLogger({
    #     f"s{i + 1}": f"logs/s{i + 1}.log"
    #     for i in range(num_of_agents)
    # }, mode='w', language="he")
    string_explanation_logger = StringsExplanationLogger([f"s{i + 1}" for i in range(num_of_agents)], level=logging.DEBUG)

    # print("\n\nIterated Maximum Matching without adjustments:")
    # divide_random_instance(algorithm=iterated_maximum_matching, adjust_utilities=False,
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12],
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)

    divide_random_instance(algorithm=SP_O_function,
                           # explanation_logger=console_explanation_logger,
                           #    explanation_logger = files_explanation_logger,
                           explanation_logger=string_explanation_logger,
                           num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2, 5],
                           item_capacity_bounds=[3, 12],
                           item_base_value_bounds=[1, 100], item_subjective_ratio_bounds=[0.5, 1.5],
                           normalized_sum_of_values=100,
                           random_seed=1)

    # print(string_explanation_logger.map_agent_to_explanation())
    print(string_explanation_logger.map_agent_to_explanation()["s1"])

