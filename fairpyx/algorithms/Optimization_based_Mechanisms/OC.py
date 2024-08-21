"""
    "Optimization-based Mechanisms for the Course Allocation Problem", by Hoda Atef Yekta, Robert Day (2020)
     https://doi.org/10.1287/ijoc.2018.0849

    Programmer: Tamar Bar-Ilan, Moriya Ester Ohayon, Ofek Kats
"""

import cvxpy
from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
import cvxpy as cp
import numpy as np
import fairpyx.algorithms.Optimization_based_Mechanisms.optimal_functions as optimal
logger = logging.getLogger(__name__)

def conflicts_condition(alloc, x, explanation_logger):
    conditions = []
    list_courses = []
    for course in alloc.remaining_items():
        list_courses.append(course)

    for course in alloc.remaining_items():
        list_of_conflict = alloc.remaining_instance().item_conflicts(course)
        explanation_logger.debug("for course %s list_of_conflict %s", course, list_of_conflict)
        for course2 in list_of_conflict:
            index_c1 = list_courses.index(course)
            index_c2 = list_courses.index(course2)
            for i in range(len(alloc.remaining_agents())):
                conditions.append(x[index_c1, i] + x[index_c2, i] <= 1)

    return conditions

def OC_function(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger(), solver=None):
    """
    Algorethem 5: Allocate the given items to the given agents using the OC protocol.

    in the OC algorithm for CAP, we maximize ordinal utility followed by maximizing cardinal utility among rank-maximal
    solutions, performing this two-part optimization once for the whole market.

    :param alloc: an allocation builder, which tracks the allocation and the remaining capacity for items and agents of
     the fair course allocation problem(CAP).

    :param solver: solver for cvxpy. Default is depend on the version.

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

    rank_mat = optimal.createRankMat(alloc,explanation_logger)
    sum_rank = optimal.sumOnRankMat(alloc, rank_mat, x)
    objective_Z1 = cp.Maximize(sum_rank)

    constraints_Z1 = optimal.notExceedtheCapacity(x,alloc) \
                     + optimal.numberOfCourses(x, alloc, alloc.remaining_agent_capacities) \
                     + conflicts_condition(alloc, x, explanation_logger)

    problem = cp.Problem(objective_Z1, constraints=constraints_Z1)
    explanation_logger.debug("solver: %s", solver)
    result_Z1 = problem.solve(solver=solver)
    explanation_logger.debug("\nRank optimization: result_Z1 = %s, x = \n%s", result_Z1, x.value)

    x = cvxpy.Variable((len(alloc.remaining_items()), len(alloc.remaining_agents())), boolean=True)  # Is there a func which zero all the matrix?
    sum_rank = optimal.sumOnRankMat(alloc, rank_mat, x)
    objective_Z2 = cp.Maximize(cp.sum([alloc.effective_value(student, course) * x[j, i]
                                        for j, course in enumerate(alloc.remaining_items())
                                        for i, student in enumerate(alloc.remaining_agents())
                                        if (student, course) not in alloc.remaining_conflicts]))

    # condition number 19:
    constraints_Z2 = optimal.notExceedtheCapacity(x, alloc) \
                     + optimal.numberOfCourses(x, alloc, alloc.remaining_agent_capacities) \
                     + conflicts_condition(alloc, x, explanation_logger)

    constraints_Z2.append(sum_rank >= int(result_Z1))


    # logger.info("type(alloc.instance.item_conflicts) = %s ", type(alloc.instance.item_conflicts))
    # explanation_logger.debug("alloc.remaining_conflicts = %s ", alloc.remaining_conflicts)
    # explanation_logger.debug("alloc.remaining_instance().item_conflicts(c1) = %s ", alloc.remaining_instance().item_conflicts("c1"))

    try:
        problem = cp.Problem(objective_Z2, constraints=constraints_Z2)
        result_Z2 = problem.solve(solver=solver) #, verbose=True)
        explanation_logger.debug("\nValue optimization: result_Z2 = %s, x = \n%s", result_Z2, x.value)

        # Check if the optimization problem was successfully solved
        if result_Z2 is not None:
            optimal.give_items_according_to_allocation_matrix(alloc, x, explanation_logger, rank_mat)

            optimal_value = problem.value
            explanation_logger.debug("Optimal Objective Value: %s", optimal_value)
            # Now you can use this optimal value for further processing
        else:
            explanation_logger.info("Solver failed to find a solution or the problem is infeasible/unbounded.")
            raise ValueError("Solver failed to find a solution or the problem is infeasible/unbounded.")

    except Exception as e:
        explanation_logger.info("Solver failed: %s", str(e))
        explanation_logger.warning("An error occurred: %s", str(e))
        raise


if __name__ == "__main__":
    import doctest, sys
    print("\n", doctest.testmod(), "\n")
    # sys.exit(1)

    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.DEBUG)
    # import fairpyx
    # from fairpyx.adaptors import divide
    #
    # valuations =  {
    #     's1': {'c1': 1, 'c2': 4, 'c3': 5, 'c4': 2, 'c5': 18},
    #     's2': {'c1': 1, 'c2': 4, 'c3': 3, 'c4': 2, 'c5': 20},
    #     's3': {'c1': 3, 'c2': 5, 'c3': 3, 'c4': 2, 'c5': 17},
    #     's4': {'c1': 2, 'c2': 1, 'c3': 6, 'c4': 2, 'c5': 19},
    #     's5': {'c1': 2, 'c2': 5, 'c3': 3, 'c4': 2, 'c5': 19}
    # }
    # agent_capacities = {'s1': 3, 's2': 3, 's3': 3, 's4': 3, 's5': 3}
    # item_capacities = {'c1': 5, 'c2': 5, 'c3': 5, 'c4': 5, 'c5': 5}
    # instance = fairpyx.Instance(valuations=valuations, agent_capacities=agent_capacities, item_capacities=item_capacities)
    # solver = None
    # allocation = fairpyx.divide(OC_function, instance=instance, solver=solver)
    # fairpyx.validate_allocation(instance, allocation, title=f"OC_function")


    # for i in range(100):
    #     np.random.seed(i)
    #     instance = fairpyx.Instance.random_uniform(
    #         num_of_agents=5, num_of_items=5, normalized_sum_of_values=30,
    #         agent_capacity_bounds=[2,6],
    #         item_capacity_bounds=[20,40],
    #         item_base_value_bounds=[1,10],
    #         item_subjective_ratio_bounds=[0.5, 1.5]
    #         )
    #     print("instance: ",instance)
    #     print("valuations: ",instance._valuations)
    #     allocation = fairpyx.divide(fairpyx.algorithms.OC_function, instance=instance)
    #     fairpyx.validate_allocation(instance, allocation, title=f"Seed {i}, OC_function")


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
    # divide(OC_function, instance=instance)

    #s1 = {"c1": 400, "c2": 150, "c3": 230, "c4": 200, "c5": 20}
    #s2 = {"c1": 245, "c2": 252, "c3": 256, "c4": 246, "c5": 1}
    #s3 = {"c1": 243, "c2": 230, "c3": 240, "c4": 245, "c5": 42}
    #s4 = {"c1": 251, "c2": 235, "c3": 242, "c4": 201, "c5": 71}
    #instance = Instance(
    #    agent_capacities={"s1": 3, "s2": 3, "s3": 3, "s4": 3},
    #    item_capacities={"c1": 2, "c2": 3, "c3": 3, "c4": 2, "c5": 2},
    #    item_conflicts={"c1": ['c4'], "c4": ['c1']},
    #    valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4}
    #)

    #divide(OC_function, instance=instance)

    # np.random.seed(2)
    # instance = fairpyx.Instance.random_uniform(
    #     num_of_agents=70, num_of_items=10, normalized_sum_of_values=100,
    #     agent_capacity_bounds=[2, 6],
    #     item_capacity_bounds=[20, 40],
    #     item_base_value_bounds=[1, 1000],
    #     item_subjective_ratio_bounds=[0.5, 1.5]
    # )
    # allocation = divide(OC_function, instance=instance)

    # s1 = {"c1": 40, "c2": 20, "c3": 10, "c4": 30}
    # s2 = {"c1": 6, "c2": 20, "c3": 70, "c4": 4}
    # s3 = {"c1": 9, "c2": 20, "c3": 21, "c4": 50}
    # s4 = {"c1": 25, "c2": 5, "c3": 15, "c4": 55}
    # s5 = {"c1": 5, "c2": 90, "c3": 3, "c4": 2}
    # instance = fairpyx.Instance(
    #     agent_capacities={"s1": 2, "s2": 2, "s3": 2, "s4": 2, "s5": 2},
    #     item_capacities={"c1": 3, "c2": 2, "c3": 2, "c4": 2},
    #     valuations={"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}
    # )
    # allocation = divide(OC_function, instance=instance)

    from fairpyx.adaptors import divide_random_instance, divide
    from fairpyx.explanations import ConsoleExplanationLogger, FilesExplanationLogger, StringsExplanationLogger

    num_of_agents = 5
    num_of_items = 3

    console_explanation_logger = ConsoleExplanationLogger(level=logging.INFO)
    # files_explanation_logger = FilesExplanationLogger({
    #     f"s{i + 1}": f"logs/s{i + 1}.log"
    #     for i in range(num_of_agents)
    # }, mode='w', language="he")
    # string_explanation_logger = StringsExplanationLogger(f"s{i + 1}" for i in range(num_of_agents))

    # print("\n\nIterated Maximum Matching without adjustments:")
    # divide_random_instance(algorithm=iterated_maximum_matching, adjust_utilities=False,
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2,5], item_capacity_bounds=[3,12],
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=100,
    #                        random_seed=1)

#################################################################console logger
    # print("\n\nIterated Maximum Matching with adjustments:")
    # divide_random_instance(algorithm=OC_function,
    #                        explanation_logger=console_explanation_logger,
    #                        #    explanation_logger = files_explanation_logger,
    #                        # explanation_logger=string_explanation_logger,
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, agent_capacity_bounds=[2, 5],
    #                        item_capacity_bounds=[3, 12],
    #                        item_base_value_bounds=[1, 100], item_subjective_ratio_bounds=[0.5, 1.5],
    #                        normalized_sum_of_values=100,
    #                        random_seed=1)

    # print(string_explanation_logger.map_agent_to_explanation())
    # print(string_explanation_logger.map_agent_to_explanation()["s1"])



    valuations= {
        "s64": {"lmydt mKHvnh": 501, "hskh sTTysTyt": 211, "SHyTvt lgylvy htkpvt syybr": 112, "rvbvTym AvTvnvmyym": 64,
                "prTyvt HySHvb": 47, "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 40,
                "tAvryh SHl krypTvgrpyh": 6, "dHyst ntvnym byvm b 9:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "spykvt bvlyAnyt": 0, "Algvrytmym bbynh mlAKHvtyt": 0},
        "s67": {"dHyst ntvnym byvm b 9:00": 170, "ptrvn b`yvt bAmTS`vt HypvSH": 0, "Algvrytmym KHlKHlyym": 166,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 166,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 166, "prTyvt HySHvb": 0, "tAvryh SHl krypTvgrpyh": 0,
                "spykvt bvlyAnyt": 166, "Algvrytmym bbynh mlAKHvtyt": 0},
        "s68": {"nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 204, "rvbvTym AvTvnvmyym": 163,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 145, "SHyTvt lgylvy htkpvt syybr": 127,
                "nytvH myd` bmymdym gbvhym": 109, "hskh sTTysTyt": 90, "lmydt mKHvnh": 72, "Algvrytmym KHlKHlyym": 54,
                "Algvrytmym bbynh mlAKHvtyt": 36, "dHyst ntvnym byvm b 9:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "gyAvmTryh bdydh": 0, "sybvKHyvt tkSHvrt": 0, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "prTyvt HySHvb": 0,
                "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0},
        "s69": {"dHyst ntvnym byvm b 9:00": 250, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 250,
                "nytvH myd` bmymdym gbvhym": 250, "ptrvn b`yvt bAmTS`vt HypvSH": 0, "Algvrytmym KHlKHlyym": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0, "prTyvt HySHvb": 0,
                "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0, "Algvrytmym bbynh mlAKHvtyt": 0},
        "s71": {"dHyst ntvnym byvm b 9:00": 288, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 249,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 201, "SHyTvt lgylvy htkpvt syybr": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 262, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s76": {"dHyst ntvnym byvm b 9:00": 0, "dHyst ntvnym byvm g 14:00": 103, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 144, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 237,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 440,
                "tKHnvt Algvrytmym mHkryym": 76, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s78": {"lmydh `mvkh v`ybvd SHpvt Tb`yvt": 600, "lmydt mKHvnh": 400,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "dHyst ntvnym byvm b 9:00": 0,
                "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0, "Algvrytmym KHlKHlyym": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "sybvKHyvt tkSHvrt": 0,
                "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0, "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0,
                "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0, "tKHnvt Algvrytmym mHkryym": 0,
                "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s80": {"nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 395, "dHyst ntvnym byvm b 9:00": 350,
                "nytvH myd` bmymdym gbvhym": 131, "Algvrytmym KHlKHlyym": 124, "hskh sTTysTyt": 0,
                "dHyst ntvnym byvm g 14:00": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "ptrvn b`yvt bAmTS`vt HypvSH": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0,
                "rvbvTym AvTvnvmyym": 0, "SHyTvt lgylvy htkpvt syybr": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0,
                "sybvKHyvt tkSHvrt": 0, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0,
                "spykvt bvlyAnyt": 0, "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0,
                "pytvH mSHHky mHSHb": 0, "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s81": {"dHyst ntvnym byvm b 9:00": 168, "dHyst ntvnym byvm g 14:00": 168, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 166,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 166, "lmydt mKHvnh": 166, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 166,
                "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s83": {"dHyst ntvnym byvm b 9:00": 280, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 40,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 5, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 103, "SHyTvt lgylvy htkpvt syybr": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 199, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 190, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 183, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s84": {"lmydt mKHvnh": 200, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 177, "pytvH mSHHky mHSHb": 159,
                "Algvrytmym bbynh mlAKHvtyt": 133, "Algvrytmym KHlKHlyym": 111, "ptrvn b`yvt bAmTS`vt HypvSH": 88,
                "dHyst ntvnym byvm b 9:00": 66, "dHyst ntvnym byvm g 14:00": 44, "rvbvTym AvTvnvmyym": 22,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0,
                "hskh sTTysTyt": 0, "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0,
                "SHyTvt lgylvy htkpvt syybr": 0, "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0,
                "sybvKHyvt tkSHvrt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0, "prTyvt HySHvb": 0,
                "mbvA lkrypTvgrpyh": 0, "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s85": {"dHyst ntvnym byvm b 9:00": 124, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 393, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 209, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 119, "prTyvt HySHvb": 88, "mbvA lkrypTvgrpyh": 67,
                "pytvH mSHHky mHSHb": 0, "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s87": {"dHyst ntvnym byvm b 9:00": 0, "dHyst ntvnym byvm g 14:00": 200, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 206,
                "nytvH myd` bmymdym gbvhym": 78, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 137,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 96, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 212,
                "tKHnvt Algvrytmym mHkryym": 71, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s88": {"dHyst ntvnym byvm b 9:00": 502, "dHyst ntvnym byvm g 14:00": 0, "lmydt mKHvnh": 498,
                "ptrvn b`yvt bAmTS`vt HypvSH": 0, "Algvrytmym KHlKHlyym": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0,
                "hskh sTTysTyt": 0, "rvbvTym AvTvnvmyym": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "sybvKHyvt tkSHvrt": 0,
                "tAvryh SHl krypTvgrpyh": 0, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s90": {"Algvrytmym KHlKHlyym": 149, "hskh sTTysTyt": 111, "dHyst ntvnym byvm g 14:00": 90,
                "dHyst ntvnym byvm b 9:00": 111, "SHyTvt lgylvy htkpvt syybr": 90, "nytvH myd` bmymdym gbvhym": 90,
                "ptrvn b`yvt bAmTS`vt HypvSH": 90, "lmydt mKHvnh": 90, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 31,
                "pytvH mSHHky mHSHb": 0, "nvSHAym mtkdmym btvrt hgrpym": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0,
                "rvbvTym AvTvnvmyym": 0, "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0,
                "gyAvmTryh bdydh": 90, "sybvKHyvt tkSHvrt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0,
                "tKHnvt Algvrytmym mHkryym": 58},
        "s91": {"dHyst ntvnym byvm b 9:00": 0, "dHyst ntvnym byvm g 14:00": 42, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 271,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 301,
                "nytvH myd` bmymdym gbvhym": 386, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s96": {"dHyst ntvnym byvm b 9:00": 0, "dHyst ntvnym byvm g 14:00": 200, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 100,
                "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 500, "sybvKHyvt tkSHvrt": 100,
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 100, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s100": {"dHyst ntvnym byvm b 9:00": 171, "dHyst ntvnym byvm g 14:00": 151, "SHyTvt lgylvy htkpvt syybr": 136,
                 "nytvH myd` bmymdym gbvhym": 121, "lmydt mKHvnh": 106, "Algvrytmym KHlKHlyym": 90,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 75, "hskh sTTysTyt": 60,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "gyAvmTryh bdydh": 0,
                 "sybvKHyvt tkSHvrt": 0, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "spykvt bvlyAnyt": 0,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 45, "mbvA lkrypTvgrpyh": 30,
                 "tAvryh SHl krypTvgrpyh": 15, "pytvH mSHHky mHSHb": 0, "tKHnvt Algvrytmym mHkryym": 0,
                 "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s101": {"dHyst ntvnym byvm b 9:00": 138, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 33,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                 "nytvH myd` bmymdym gbvhym": 15, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 478, "sybvKHyvt tkSHvrt": 120,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 105, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 13, "mbvA lkrypTvgrpyh": 88, "pytvH mSHHky mHSHb": 0,
                 "tKHnvt Algvrytmym mHkryym": 10, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s106": {"dHyst ntvnym byvm b 9:00": 502, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 498, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 0,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                 "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                 "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s108": {"dHyst ntvnym byvm b 9:00": 322, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "Algvrytmym KHlKHlyym": 145, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 184,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 180,
                 "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 67,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 102, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                 "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s109": {"pytvH mSHHky mHSHb": 175, "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 175, "hskh sTTysTyt": 172,
                 "Algvrytmym KHlKHlyym": 150, "dHyst ntvnym byvm g 14:00": 0, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "rvbvTym AvTvnvmyym": 0,
                 "dHyst ntvnym byvm b 9:00": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 178, "nytvH myd` bmymdym gbvhym": 150,
                 "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "sybvKHyvt tkSHvrt": 0,
                 "tAvryh SHl krypTvgrpyh": 0, "Algvrytmym bbynh mlAKHvtyt": 0, "spykvt bvlyAnyt": 0, "prTyvt HySHvb": 0,
                 "mbvA lkrypTvgrpyh": 0, "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s110": {"dHyst ntvnym byvm g 14:00": 244, "dHyst ntvnym byvm b 9:00": 74, "SHyTvt lgylvy htkpvt syybr": 230,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 243, "lmydt mKHvnh": 145, "nytvH myd` bmymdym gbvhym": 0,
                 "prTyvt HySHvb": 64, "Algvrytmym KHlKHlyym": 0, "tKHnvt Algvrytmym mHkryym": 0,
                 "ptrvn b`yvt bAmTS`vt HypvSH": 0, "hskh sTTysTyt": 0, "gyAvmTryh bdydh": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0,
                 "rvbvTym AvTvnvmyym": 0, "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0,
                 "sybvKHyvt tkSHvrt": 0, "tAvryh SHl krypTvgrpyh": 0, "Algvrytmym bbynh mlAKHvtyt": 0,
                 "spykvt bvlyAnyt": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                 "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s111": {"dHyst ntvnym byvm b 9:00": 157, "dHyst ntvnym byvm g 14:00": 139, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 165, "hskh sTTysTyt": 132,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 407,
                 "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 0, "sybvKHyvt tkSHvrt": 0,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                 "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s112": {"dHyst ntvnym byvm b 9:00": 120, "dHyst ntvnym byvm g 14:00": 92, "ptrvn b`yvt bAmTS`vt HypvSH": 0,
                 "Algvrytmym KHlKHlyym": 0, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 320,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 356,
                 "nytvH myd` bmymdym gbvhym": 0, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 112, "sybvKHyvt tkSHvrt": 0,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                 "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0},
        "s113": {"dHyst ntvnym byvm b 9:00": 200, "dHyst ntvnym byvm g 14:00": 200, "ptrvn b`yvt bAmTS`vt HypvSH": 100,
                 "Algvrytmym KHlKHlyym": 160, "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 0,
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 0, "rvbvTym AvTvnvmyym": 0, "hskh sTTysTyt": 200,
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 0, "SHyTvt lgylvy htkpvt syybr": 0,
                 "nytvH myd` bmymdym gbvhym": 40, "gyAvmTryh bdydh": 0, "lmydt mKHvnh": 100, "sybvKHyvt tkSHvrt": 0,
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 0, "tAvryh SHl krypTvgrpyh": 0, "spykvt bvlyAnyt": 0,
                 "Algvrytmym bbynh mlAKHvtyt": 0, "prTyvt HySHvb": 0, "mbvA lkrypTvgrpyh": 0, "pytvH mSHHky mHSHb": 0,
                 "tKHnvt Algvrytmym mHkryym": 0, "nvSHAym mtkdmym btvrt hgrpym": 0}}
    agent_capacities= {"s64": 6, "s67": 6, "s68": 4, "s69": 4, "s71": 6, "s76": 6, "s78": 6, "s80": 2, "s81": 6,
                          "s83": 6, "s84": 6, "s85": 6, "s87": 2, "s88": 6, "s90": 5, "s91": 3, "s96": 6, "s100": 6,
                          "s101": 6, "s106": 6, "s108": 6, "s109": 6, "s110": 3, "s111": 6, "s112": 4, "s113": 5}
    item_capacities= {"dHyst ntvnym byvm b 9:00": 126, "dHyst ntvnym byvm g 14:00": 40,
                         "ptrvn b`yvt bAmTS`vt HypvSH": 82, "Algvrytmym KHlKHlyym": 44,
                         "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": 50,
                         "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": 50, "rvbvTym AvTvnvmyym": 69,
                         "hskh sTTysTyt": 80, "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt": 40,
                         "SHyTvt lgylvy htkpvt syybr": 55, "nytvH myd` bmymdym gbvhym": 77, "gyAvmTryh bdydh": 67,
                         "lmydt mKHvnh": 81, "sybvKHyvt tkSHvrt": 50, "lmydh `mvkh v`ybvd SHpvt Tb`yvt": 86,
                         "tAvryh SHl krypTvgrpyh": 50, "spykvt bvlyAnyt": 40, "Algvrytmym bbynh mlAKHvtyt": 84,
                         "prTyvt HySHvb": 65, "mbvA lkrypTvgrpyh": 50, "pytvH mSHHky mHSHb": 40,
                         "tKHnvt Algvrytmym mHkryym": 41, "nvSHAym mtkdmym btvrt hgrpym": 40}
    agent_conflicts= {
        "s71": ["SHyTvt lgylvy htkpvt syybr", "gyAvmTryh bdydh", "nytvH myd` bmymdym gbvhym",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "spykvt bvlyAnyt",
                "mbvA lkrypTvgrpyh", "prTyvt HySHvb", "Algvrytmym bbynh mlAKHvtyt", "sybvKHyvt tkSHvrt",
                "tAvryh SHl krypTvgrpyh"], "s76": ["tAvryh SHl krypTvgrpyh", "gyAvmTryh bdydh",
                                                   "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt",
                                                   "lmydt mKHvnh", "nytvH myd` bmymdym gbvhym",
                                                   "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00",
                                                   "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00",
                                                   "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "hskh sTTysTyt",
                                                   "dHyst ntvnym byvm b 9:00", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh",
                                                   "prTyvt HySHvb", "Algvrytmym bbynh mlAKHvtyt",
                                                   "nvSHAym mtkdmym btvrt hgrpym", "Algvrytmym KHlKHlyym",
                                                   "sybvKHyvt tkSHvrt", "ptrvn b`yvt bAmTS`vt HypvSH"],
        "s78": ["SHyTvt lgylvy htkpvt syybr", "dHyst ntvnym byvm g 14:00", "mbvA lkrypTvgrpyh",
                "tKHnvt Algvrytmym mHkryym", "nvSHAym mtkdmym btvrt hgrpym", "gyAvmTryh bdydh",
                "nytvH myd` bmymdym gbvhym", "dHyst ntvnym byvm b 9:00", "ptrvn b`yvt bAmTS`vt HypvSH",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00",
                "hskh sTTysTyt", "spykvt bvlyAnyt", "prTyvt HySHvb",
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt", "rvbvTym AvTvnvmyym", "pytvH mSHHky mHSHb",
                "Algvrytmym bbynh mlAKHvtyt", "Algvrytmym KHlKHlyym", "sybvKHyvt tkSHvrt", "tAvryh SHl krypTvgrpyh"],
        "s80": ["rvbvTym AvTvnvmyym", "tAvryh SHl krypTvgrpyh", "SHyTvt lgylvy htkpvt syybr", "gyAvmTryh bdydh",
                "lmydt mKHvnh", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "pytvH mSHHky mHSHb",
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "tKHnvt Algvrytmym mHkryym", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh",
                "prTyvt HySHvb", "Algvrytmym bbynh mlAKHvtyt", "nvSHAym mtkdmym btvrt hgrpym", "sybvKHyvt tkSHvrt",
                "ptrvn b`yvt bAmTS`vt HypvSH"], "s81": ["rvbvTym AvTvnvmyym", "tAvryh SHl krypTvgrpyh",
                                                        "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt",
                                                        "nytvH myd` bmymdym gbvhym",
                                                        "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00",
                                                        "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00",
                                                        "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "hskh sTTysTyt",
                                                        "spykvt bvlyAnyt", "tKHnvt Algvrytmym mHkryym",
                                                        "mbvA lkrypTvgrpyh", "prTyvt HySHvb",
                                                        "Algvrytmym bbynh mlAKHvtyt", "nvSHAym mtkdmym btvrt hgrpym",
                                                        "Algvrytmym KHlKHlyym", "sybvKHyvt tkSHvrt",
                                                        "ptrvn b`yvt bAmTS`vt HypvSH"],
        "s83": ["rvbvTym AvTvnvmyym", "SHyTvt lgylvy htkpvt syybr", "gyAvmTryh bdydh", "nytvH myd` bmymdym gbvhym",
                "pytvH mSHHky mHSHb", "hskh sTTysTyt", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh", "prTyvt HySHvb",
                "tKHnvt Algvrytmym mHkryym", "Algvrytmym KHlKHlyym", "nvSHAym mtkdmym btvrt hgrpym",
                "tAvryh SHl krypTvgrpyh"],
        "s84": ["tAvryh SHl krypTvgrpyh", "SHyTvt lgylvy htkpvt syybr", "gyAvmTryh bdydh", "nytvH myd` bmymdym gbvhym",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00",
                "hskh sTTysTyt", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh", "prTyvt HySHvb", "tKHnvt Algvrytmym mHkryym",
                "nvSHAym mtkdmym btvrt hgrpym", "sybvKHyvt tkSHvrt",
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s85": ["rvbvTym AvTvnvmyym", "SHyTvt lgylvy htkpvt syybr", "gyAvmTryh bdydh",
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt", "dHyst ntvnym byvm g 14:00",
                "nytvH myd` bmymdym gbvhym", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "pytvH mSHHky mHSHb",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "hskh sTTysTyt",
                "spykvt bvlyAnyt", "tKHnvt Algvrytmym mHkryym", "nvSHAym mtkdmym btvrt hgrpym", "Algvrytmym KHlKHlyym",
                "sybvKHyvt tkSHvrt", "ptrvn b`yvt bAmTS`vt HypvSH"],
        "s87": ["rvbvTym AvTvnvmyym", "gyAvmTryh bdydh", "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt",
                "lmydt mKHvnh", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "hskh sTTysTyt",
                "dHyst ntvnym byvm b 9:00", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh", "prTyvt HySHvb",
                "Algvrytmym bbynh mlAKHvtyt", "Algvrytmym KHlKHlyym", "nvSHAym mtkdmym btvrt hgrpym",
                "ptrvn b`yvt bAmTS`vt HypvSH"],
        "s88": ["SHyTvt lgylvy htkpvt syybr", "mbvA lkrypTvgrpyh", "tKHnvt Algvrytmym mHkryym",
                "nvSHAym mtkdmym btvrt hgrpym", "gyAvmTryh bdydh", "nytvH myd` bmymdym gbvhym",
                "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "ptrvn b`yvt bAmTS`vt HypvSH",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00",
                "hskh sTTysTyt", "spykvt bvlyAnyt", "prTyvt HySHvb",
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt", "rvbvTym AvTvnvmyym", "pytvH mSHHky mHSHb",
                "Algvrytmym bbynh mlAKHvtyt", "Algvrytmym KHlKHlyym", "sybvKHyvt tkSHvrt", "tAvryh SHl krypTvgrpyh"],
        "s90": ["rvbvTym AvTvnvmyym", "tAvryh SHl krypTvgrpyh", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00",
                "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh", "prTyvt HySHvb",
                "Algvrytmym bbynh mlAKHvtyt", "sybvKHyvt tkSHvrt",
                "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s96": ["rvbvTym AvTvnvmyym", "gyAvmTryh bdydh", "nytvH myd` bmymdym gbvhym", "hskh sTTysTyt",
                "dHyst ntvnym byvm b 9:00", "spykvt bvlyAnyt", "tKHnvt Algvrytmym mHkryym", "Algvrytmym KHlKHlyym",
                "nvSHAym mtkdmym btvrt hgrpym", "tAvryh SHl krypTvgrpyh"],
        "s100": ["gyAvmTryh bdydh", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "pytvH mSHHky mHSHb",
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "ptrvn b`yvt bAmTS`vt HypvSH",
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "spykvt bvlyAnyt", "tKHnvt Algvrytmym mHkryym",
                 "Algvrytmym bbynh mlAKHvtyt", "nvSHAym mtkdmym btvrt hgrpym", "sybvKHyvt tkSHvrt",
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s101": ["lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00",
                 "spykvt bvlyAnyt", "nvSHAym mtkdmym btvrt hgrpym", "ptrvn b`yvt bAmTS`vt HypvSH"],
        "s108": ["rvbvTym AvTvnvmyym", "gyAvmTryh bdydh", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00",
                 "pytvH mSHHky mHSHb", "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "ptrvn b`yvt bAmTS`vt HypvSH",
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "tKHnvt Algvrytmym mHkryym", "mbvA lkrypTvgrpyh",
                 "Algvrytmym bbynh mlAKHvtyt", "nvSHAym mtkdmym btvrt hgrpym", "sybvKHyvt tkSHvrt",
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s109": ["lmydh `mvkh v`ybvd SHpvt Tb`yvt", "nvSHAym mtkdmym btvrt hgrpym", "gyAvmTryh bdydh",
                 "sybvKHyvt tkSHvrt"],
        "s110": ["rvbvTym AvTvnvmyym", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "pytvH mSHHky mHSHb",
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "spykvt bvlyAnyt", "nvSHAym mtkdmym btvrt hgrpym",
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s111": ["tAvryh SHl krypTvgrpyh", "gyAvmTryh bdydh", "lmydt mKHvnh", "nytvH myd` bmymdym gbvhym",
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "pytvH mSHHky mHSHb",
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "ptrvn b`yvt bAmTS`vt HypvSH",
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "spykvt bvlyAnyt", "tKHnvt Algvrytmym mHkryym", "mbvA lkrypTvgrpyh",
                 "prTyvt HySHvb", "Algvrytmym bbynh mlAKHvtyt", "nvSHAym mtkdmym btvrt hgrpym", "Algvrytmym KHlKHlyym",
                 "sybvKHyvt tkSHvrt", "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s112": ["tAvryh SHl krypTvgrpyh", "gyAvmTryh bdydh", "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00",
                 "pytvH mSHHky mHSHb", "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "ptrvn b`yvt bAmTS`vt HypvSH",
                 "lmydh `mvkh v`ybvd SHpvt Tb`yvt", "spykvt bvlyAnyt", "tKHnvt Algvrytmym mHkryym", "mbvA lkrypTvgrpyh",
                 "prTyvt HySHvb", "Algvrytmym bbynh mlAKHvtyt", "nvSHAym mtkdmym btvrt hgrpym", "Algvrytmym KHlKHlyym",
                 "sybvKHyvt tkSHvrt", "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"],
        "s113": ["rvbvTym AvTvnvmyym", "tAvryh SHl krypTvgrpyh", "gyAvmTryh bdydh",
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00", "pytvH mSHHky mHSHb",
                 "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00", "lmydh `mvkh v`ybvd SHpvt Tb`yvt",
                 "tKHnvt Algvrytmym mHkryym", "spykvt bvlyAnyt", "mbvA lkrypTvgrpyh", "Algvrytmym bbynh mlAKHvtyt",
                 "nvSHAym mtkdmym btvrt hgrpym", "sybvKHyvt tkSHvrt",
                 "nvSHAym mtkdmym brAyyh mmvHSHbt `m yySHvmym bhdmyh rpvAyt"]}
    item_conflicts= {"dHyst ntvnym byvm b 9:00": ["dHyst ntvnym byvm g 14:00", "nvSHAym mtkdmym btvrt hgrpym"],
                        "dHyst ntvnym byvm g 14:00": ["dHyst ntvnym byvm b 9:00", "mbvA lkrypTvgrpyh"],
                        "nvSHAym mtkdmym btvrt hgrpym": ["hskh sTTysTyt", "dHyst ntvnym byvm b 9:00"],
                        "lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00": ["lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00"],
                        "lmydh yySHvmyt brAyyh mmvHSHbt byvm h 15:00": ["lmydh yySHvmyt brAyyh mmvHSHbt byvm d 15:00"],
                        "hskh sTTysTyt": ["nvSHAym mtkdmym btvrt hgrpym"],
                        "SHyTvt lgylvy htkpvt syybr": ["lmydt mKHvnh"], "lmydt mKHvnh": ["SHyTvt lgylvy htkpvt syybr"],
                        "mbvA lkrypTvgrpyh": ["dHyst ntvnym byvm g 14:00"]}

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    import fairpyx
    from fairpyx.adaptors import divide

    instance = fairpyx.Instance(
        agent_capacities=agent_capacities,
        item_capacities=item_capacities,
        valuations=valuations,
        item_conflicts=item_conflicts
    )
    allocation = divide(OC_function, instance=instance, solver=None, explanation_logger=console_explanation_logger)


    # import experiments_csv
    # import experiments.compare_new_algo_for_course_allocation as compare_algos
    # input_ranges_specific_solver = {
    #     "max_total_agent_capacity": [1000],
    #     "algorithm": OC_function,
    #     "random_seed": 9,
    #     "solver": [None, cp.CBC, cp.MOSEK, cp.SCIP],  # , cp.XPRESS, cp.COPT, cp.CPLEX, cp.GUROBI
    # }
    #
    # experiments_csv.Experiment.run_with_time_limit(compare_algos.course_allocation_with_random_instance_sample, input_ranges_specific_solver,
    #                                time_limit=300)

