"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""
import cvxpy
import numpy as np
import fairpyx
import fairpyx.algorithms.Optimization_based_Mechanisms.optimal_functions as optimal

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
import cvxpy as cp
logger = logging.getLogger(__name__)



def TTC_O_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorethem 3: Allocate the given items to the given agents using the TTC-O protocol.

    TTC-O assigns one course in each round to each student, the winning students are defined based on
     the students’ bid values. Uses linear planning for optimality.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).


    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 50, "c2": 49, "c3": 1}
    >>> s2 = {"c1": 48, "c2": 46, "c3": 6}
    >>> agent_capacities = {"s1": 1, "s2": 1}                                 # 2 seats required
    >>> course_capacities = {"c1": 1, "c2": 1, "c3": 1}                       # 3 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(TTC_O_function, instance=instance)
    {'s1': ['c2'], 's2': ['c1']}
    """
    explanation_logger.info("\nAlgorithm TTC-O starts.\n")

    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    logger.info("Max iterations: %d", max_iterations)
    for iteration in range(max_iterations):
        logger.info("Iteration number: %d", iteration+1)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            logger.info("There are no more agents (%d) or items(%d) ", len(alloc.remaining_agent_capacities),len(alloc.remaining_item_capacities))
            break

        result_Zt1, result_Zt2, var, problem , rank_mat= optimal.roundTTC_O(alloc, logger, alloc.effective_value, 0)

        # Check if the optimization problem was successfully solved
        if result_Zt2 is not None:
            optimal.allocations(alloc, var, logger)

            optimal_value = problem.value
            explanation_logger.info("Optimal Objective Value:", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            explanation_logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")


if __name__ == "__main__":
    import doctest
    print(doctest.testmod())