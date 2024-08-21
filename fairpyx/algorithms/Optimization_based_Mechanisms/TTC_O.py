import cvxpy
import numpy as np
import fairpyx
import fairpyx.algorithms.Optimization_based_Mechanisms.optimal_functions as optimal
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
import cvxpy as cp

logger = logging.getLogger(__name__)

# if flag_if_use_alloc_in_func == 0 then using alloc.effective_value for TTC-O
# if flag_if_use_alloc_in_func == 1 then using effective_value_with_price for SP-O
def roundTTC_O(alloc, explanation_logger, agent_item_value_func, flag_if_use_alloc_in_func, rank_mat, solver=None):
    # rank_mat = optimal.createRankMat(alloc, logger)

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)

    sum_rank = optimal.sumOnRankMat(alloc, rank_mat, x)

    objective_Zt1 = cp.Maximize(sum_rank)

    constraints_Zt1 = optimal.notExceedtheCapacity(x, alloc) + optimal.numberOfCourses(x, alloc, 1)

    problem = cp.Problem(objective_Zt1, constraints=constraints_Zt1)
    explanation_logger.debug("solver : %s", solver)
    result_Zt1 = problem.solve(solver=solver)  # This is the optimal value of program (6)(7)(8)(9).
    explanation_logger.debug("result_Zt1 - the optimum ranking: %d", result_Zt1)

    # Write and solve new program for Zt2 (10)(11)(7)(8)
    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)
    sum_rank = optimal.sumOnRankMat(alloc, rank_mat, x)
    explanation_logger.debug("flag_if_use_alloc_in_func(0=TTC-O, 1=SP-O): %d", flag_if_use_alloc_in_func)
    objective_Zt2 = cp.Maximize(cp.sum(
        [agent_item_value_func(student, course) * x[j, i] if flag_if_use_alloc_in_func == 0 else agent_item_value_func(alloc, student, course) * x[j, i]
         for j, course in enumerate(alloc.remaining_items())
         for i, student in enumerate(alloc.remaining_agents())
         if (student, course) not in alloc.remaining_conflicts]))

    constraints_Zt2 = optimal.notExceedtheCapacity(x, alloc) + optimal.numberOfCourses(x, alloc, 1)

    constraints_Zt2.append(sum_rank >= int(result_Zt1))

    try:
        problem = cp.Problem(objective_Zt2, constraints=constraints_Zt2)
        result_Zt2 = problem.solve(solver=solver)
        explanation_logger.debug("result_Zt2 - the optimum bids: %d", result_Zt2)

    except Exception as e:
        explanation_logger.info("Solver failed: %s", str(e))
        explanation_logger.error("An error occurred: %s", str(e))
        raise

    return result_Zt1, result_Zt2, x, problem

def TTC_O_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger(), solver=None):
    """
    Algorithm 3: Allocate the given items to the given agents using the TTC-O protocol.

    TTC-O assigns one course in each round to each student, the winning students are defined based on
     the students’ bid values. Uses linear planning for optimality.

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
    >>> divide(TTC_O_function, instance=instance)
    {'s1': ['c2'], 's2': ['c1']}
    """
    explanation_logger.info("\nAlgorithm TTC-O starts.\n")
    all_agents = set(alloc.remaining_agents())
    max_iterations = max(alloc.remaining_agent_capacities[agent] for agent in alloc.remaining_agents())  # the amount of courses of student with maximum needed courses
    explanation_logger.debug("Max iterations: %d", max_iterations)

    rank_mat = optimal.createRankMat(alloc, explanation_logger)
    for iteration in range(max_iterations):
        explanation_logger.info("\nIteration number: %d", iteration+1)
        agents_who_need_an_item_in_current_iteration = set(alloc.remaining_agents())  # only the agents that still need courses

        # Find the difference between all_agents and agents_who_need_an_item_in_current_iteration
        agents_not_in_need = all_agents - agents_who_need_an_item_in_current_iteration
        # If you need the result as a list:
        agents_not_in_need_list = list(agents_not_in_need)
        for student in agents_not_in_need_list:
            explanation_logger.info("There are no more items you can get", agents=student)
        if len(alloc.remaining_agent_capacities) == 0 or len(alloc.remaining_item_capacities) == 0:  # check if all the agents got their courses or there are no more
            explanation_logger.info("There are no more agents (%d) or items (%d): algorithm ends", len(alloc.remaining_agent_capacities),len(alloc.remaining_item_capacities))
            break

        result_Zt1, result_Zt2, var, problem = roundTTC_O(alloc, explanation_logger, alloc.effective_value, 0, rank_mat, solver)

        # Check if the optimization problem was successfully solved
        if result_Zt2 is not None:
            rank_mat = optimal.give_items_according_to_allocation_matrix(alloc, var, explanation_logger, rank_mat)

            optimal_value = problem.value
            explanation_logger.debug("Optimal Objective Value: %s", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            explanation_logger.debug("Solver failed to find a solution or the problem is infeasible/unbounded.")

if __name__ == "__main__":
    import doctest, sys, numpy as np
    print("\n", doctest.testmod(), "\n")
    # sys.exit(1)

    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.INFO)

    # from fairpyx.adaptors import divide
    #
    # np.random.seed(2)
    # instance = fairpyx.Instance.random_uniform(
    #     num_of_agents=70, num_of_items=10, normalized_sum_of_values=100,
    #     agent_capacity_bounds=[2, 6],
    #     item_capacity_bounds=[20, 40],
    #     item_base_value_bounds=[1, 1000],
    #     item_subjective_ratio_bounds=[0.5, 1.5]
    # )
    # solver = None
    # allocation = divide(TTC_O_function, instance=instance, solver=solver)
    # fairpyx.validate_allocation(instance, allocation, title=f"Seed {5}, TTC_O_function")

    from fairpyx.adaptors import divide_random_instance, divide
    from fairpyx.explanations import ConsoleExplanationLogger, FilesExplanationLogger, StringsExplanationLogger

    num_of_agents = 5
    num_of_items = 3

    # console_explanation_logger = ConsoleExplanationLogger(level=logging.INFO)
    # files_explanation_logger = FilesExplanationLogger({
    #     f"s{i + 1}": f"logs/s{i + 1}.log"
    #     for i in range(num_of_agents)
    # }, mode='w', language="he")
    string_explanation_logger = StringsExplanationLogger([f"s{i + 1}" for i in range(num_of_agents)], level=logging.INFO)

    # print("\n\nIterated Maximum Matching without adjustments:")
    # divide_random_instance(algorithm=iterated_maximum_matching, adjust_utilities=False,
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12],
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)

    print("\n\nIterated Maximum Matching with adjustments:")
    divide_random_instance(algorithm=TTC_O_function,
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


