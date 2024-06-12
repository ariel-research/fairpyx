"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""

import cvxpy

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
import cvxpy as cp
import fairpyx.algorithms.Optimization_based_Mechanisms.optimal_functions as optimal
logger = logging.getLogger(__name__)


def OC_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorethem 5: Allocate the given items to the given agents using the OC protocol.

    in the OC algorithm for CAP, we maximize ordinal utility followed by maximizing cardinal utility among rank-maximal
    solutions, performing this two-part optimization once for the whole market.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 44, "c2": 39, "c3": 17}
    >>> s2 = {"c1": 50, "c2": 45, "c3": 5}
    >>> agent_capacities = {"s1": 2, "s2": 2}                                 # 4 seats required
    >>> course_capacities = {"c1": 2, "c2": 1, "c3": 2}                       # 5 seats available
    >>> valuations = {"s1": s1, "s2": s2}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(OC_function, instance=instance)
    {'s1': ['c1', 'c3'], 's2': ['c1', 'c2']}
    """

    explanation_logger.info("\nAlgorithm OC starts.\n")

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)

    rank_mat = optimal.createRankMat(alloc,logger)

    sum_rank = optimal.sumOnRankMat(alloc, rank_mat, x)

    objective_Z1 = cp.Maximize(sum_rank)

    constraints_Z1 = optimal.notExceedtheCapacity(x,alloc) + optimal.numberOfCourses(x, alloc, alloc.remaining_agent_capacities)

    problem = cp.Problem(objective_Z1, constraints=constraints_Z1)
    result_Z1 = problem.solve()
    logger.info("result_Z1 - the optimum ranking: %d", result_Z1)

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)  # Is there a func which zero all the matrix?

    objective_Z2 = cp.Maximize(cp.sum([alloc.effective_value(student, course) * x[j, i]
                                        for j, course in enumerate(alloc.remaining_items())
                                        for i, student in enumerate(alloc.remaining_agents())
                                        if (student, course) not in alloc.remaining_conflicts]))

    # condition number 19:
    constraints_Z2 = optimal.notExceedtheCapacity(x, alloc) + optimal.numberOfCourses(x, alloc, alloc.remaining_agent_capacities)

    constraints_Z2.append(sum_rank == result_Z1)

    try:
        problem = cp.Problem(objective_Z2, constraints=constraints_Z2)
        result_Z2 = problem.solve()
        logger.info("result_Z2 - the optimum bids: %d", result_Z2)

        # Check if the optimization problem was successfully solved
        if result_Z2 is not None:
            optimal.allocations(alloc, x, logger)

            optimal_value = problem.value
            explanation_logger.info("Optimal Objective Value:", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            explanation_logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")
            raise ValueError("Solver failed to find a solution or the problem is infeasible/unbounded.")

    except Exception as e:
        explanation_logger.info("Solver failed: %s", str(e))
        logger.error("An error occurred: %s", str(e))
        raise

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

