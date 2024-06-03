"""
An implementation of the algorithms in:
"Maximin-Aware Allocations of Indivisible Goods" by H. Chan, J. Chen, B. Li, and X. Wu (2019)
https://arxiv.org/abs/1905.09969
Programmer: Sonya Rybakov
Since: 2024-05
 # TODO: add loggers to assist funcs
 # TODO: objectify the algorithms?
 # TODO: errors for bad input
Disclaimer: all algorithms are on additive valuations
 """
import logging

import networkz as nx
from prtpy import partition, objectives as obj, outputtypes as out
from prtpy.partitioning import integer_programming

from fairpyx import Instance, AllocationBuilder, divide, ExplanationLogger, AgentBundleValueMatrix, \
    ConsoleExplanationLogger

logger = logging.getLogger(__name__)


def check_value(instance: Instance, algo_prefix: str):
    for item in instance.items:
        if instance.item_capacity(item) != 1:
            raise ValueError(algo_prefix + "item capacity restricted to only one")

    for agent in instance.agents:
        if instance.agent_capacity(agent) < instance.num_of_items:
            raise ValueError(algo_prefix + "agent capacity should be as many as the items")


def leximin_partition(valuation: dict, n: int = 3, result: out.OutputType = out.Partition):
    """
    A leximin n-partition is a partition which divides the items into n subsets and
    maximizes the lexicographical order when the values of the partitions are sorted in non-decreasing
    order. In other words, it maximizes the minimum value over all possible n partitions,
    and if there is a tie it selects the one maximizing the second minimum value, and so on

    :param valuation: a dictionary to represent the items and their valuations
    :param n: the the number of subsets
    :param result: the desired result, default is a partition represented by item keys,
    anything else is for testing and debugging purposes

    >>> print(leximin_partition({0:9,1:10}))
    [[], [0], [1]]

    >>> leximin_partition({0:2,1:8,2:8,3:7} ,result=out.PartitionAndSumsTuple)
    (array([8., 8., 9.]), [[1], [2], [0, 3]])

    >>> leximin_partition({0:2,1:8,2:8,3:7,4:5,5:2,6:3}, result=out.PartitionAndSumsTuple)
    (array([11., 11., 13.]), [[2, 6], [0, 3, 5], [1, 4]])

    >>> leximin_partition({0:2,1:2,2:2,3:2}, result=out.PartitionAndSumsTuple)
    (array([2., 2., 4.]), [[2], [1], [0, 3]])
    """
    prt = partition(algorithm=integer_programming.optimal, numbins=n, items=valuation, outputtype=result,
                    objective=obj.MaximizeSmallestSum)
    return prt


def get_bundle_rankings(instance: Instance, agent, bundles: list) -> list:
    """
    checks the ranking of bundles according to agent's preferences
    :param instance: the current instance
    :param agent: the questioned agent
    :param bundles: a list of bundles
    :return: the bundle list  sorted according to the ranking
    >>> inst1 = Instance(valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    >>> get_bundle_rankings(instance=inst1, agent='Alice', bundles=[[], [0], [1]])
    [[1], [0], []]

    >>> get_bundle_rankings(instance=inst1, agent='Bob', bundles=[[], [0], [1]])
    [[0], [1], []]

    >>> inst2 = Instance(valuations={"Alice": [2,2,6,7], "Bob": [5,3,7,5], "Claire":[2,2,2,2]})
    >>> rank_a = get_bundle_rankings(instance=inst2, agent='Alice', bundles=[[2], [1], [0, 3]])
    >>> rank_b = get_bundle_rankings(instance=inst2, agent='Bob', bundles=[[2], [1], [0, 3]])
    >>> rank_a == rank_b    # instance that goes to step 4 has same ranking
    True
    """
    ranking = sorted(bundles, key=lambda bundle: instance.agent_bundle_value(agent, bundle), reverse=True)
    return ranking


def is_significant_2nd_bundle(instance: Instance, agent, bundles: list) -> bool:
    """
    checks the significance of the second priority bundle of an agent

    :param instance: the current instance
    :param agent: the agent in question
    :param bundles: a dict which maps a bundle and its items, sorted by priorities
    :return: if the second bundle has significant value
    >>> inst2 = Instance(valuations={"Alice": [2,2,6,7], "Bob": [5,7,3,5], "Claire":[2,2,2,2]})
    >>> is_significant_2nd_bundle(instance=inst2, agent='Alice', bundles=[[0, 3], [2], [1]])
    True
    >>> is_significant_2nd_bundle(instance=inst2, agent='Bob', bundles=[[0, 3], [1], [2]])
    True
    """
    return (instance.agent_bundle_value(agent, bundles[1]) * 2) > instance.agent_bundle_value(agent, bundles[0]) \
           + instance.agent_bundle_value(agent, bundles[2])


def divide_and_choose_for_three(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorithm 1: Finds an mma1 allocation for 3 agents using leximin-n-partition.
    note: Only valuations are needed

    Examples:
    step 2 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    {'Alice': [1], 'Bob': [0], 'Claire': []}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,8,8,7,5,2,3]}
    >>> divide(divide_and_choose_for_three, valuations=val_7items)
    {'Alice': [0, 3, 5], 'Bob': [2, 6], 'Claire': [1, 4]}

    step 3 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [0], 'Bob': [2, 3], 'Claire': [1]}

    step 4-I allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,2,6,7], "Bob": [5,7,3,5], "Claire":[2,2,2,2]})
    {'Alice': [2], 'Bob': [1], 'Claire': [0, 3]}

    step 4-II allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,4,6,7], "Bob": [5,7,3,5], "Claire":[2,2,2,2]})
    {'Alice': [3], 'Bob': [0, 2], 'Claire': [1]}

    step 4-II allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [5,7,3,5], "Bob": [2,4,6,7], "Claire":[2,2,2,2]})
    {'Alice': [0, 2], 'Bob': [3], 'Claire': [1]}
    """
    DnC = {
        "algorithm_starts": {
            "he": "אלגוריתם חלק ובחר לשלושה סוכנים מתחיל",
            "en": "Divide-and-Choose Algorithm for Three Agents starts\n",
        },
        "leximin_partition": {
            "he": "%s מחלק את הפריטים %s סוכן \n",
            "en": "Agent %s divides the the items: %s \n",
        },
        "partition_results": {
            "he": " %d-חלוקה: %s \n",
            "en": "%d-partition results: %s \n",
        },
        "calc_priorities": {
            "he": "חישוב חשיבויות לסוכנים: %s, %s \n",
            "en": "Calculate priorities for the first two agents: %s, %s \n",
        },
        "first_priority_check": {
            "he": "בדיקת חשיבות ראשונה לסוכנים: %s, %s. \n",
            "en": "Checking first priorities for agents: %s, %s \n",
        },
        "different_first_priority": {
            "he": " עדיפות ראשונה שונה בין הסוכנים: %s, %s, כל אחד מקבל את העדיפות הראשונה שלו",
            "en": "agent %s’s and agent %s’s favorite bundles are different, each of them takes their favorite bundle ",
        },
        "same_first_priority": {
            "he": " עדיפות ראשונה שווה בין הסוכנים: %s, %s",
            "en": "agent %s’s and agent %s’s favorite bundles are the same",
        },
        "second_priority_check": {
            "he": "בדיקת חשיבות שנייה לסוכנים: %s, %s \n",
            "en": "Checking second priorities for agents: %s, %s \n",
        },
        "same_second_priority": {
            "he": " עדיפות שנייה שווה בין הסוכנים, ",
            "en": "Second favorite bundles are the same, ",
        },
        "given_choice": {
            "he": "הסוכן %s מקבל את זכות הבחירה.",
            "en": "Agent %s gets to choose their favorite from repartition",
        },
        "remainder": {
            "he": "החבילה הנותרת הולכת לסוכן %s \n",
            "en": "the remaining bundle is allocated to %s \n",
        },
        "different_second_priority": {
            "he": " עדיפות שנייה שווה בין הסוכנים, בדיקת ערך משמעותי עבור עדיפות שנייה",
            "en": "Second favorite bundles are different, checking significant worth of second favorites",
        },
        "has_significant_worth": {
            "he": "יש ערך משמעותי לעדיפות השנייה עבור שני הסוכנים, העדיפות השנייה מוקצית בהתאם \n",
            "en": "There is significant worth of second favorites for both. Assigning second favorites accordingly\n",
        },
        "not_significant": {
            "he": "אין ערך משמעותי עבור הסוכן %s. \n",
            "en": "There is no significant worth of second favorites for agent %s. "
        },
        "final_allocation": {
            "he": "החבילה הנותרת מוקצית לסוכן %s. \n",
            "en": "the remaining bundle is allocated to %s \n",
        },
        "repartition": {
            "he": "הסוכן %s לוקח את 2 החבילות בעדיפות עליונה לחלוקה מחדש.",
            "en": "Agent %s takes their 2 favorites for repartition \n",
        },

    }

    def _(code: str):
        return DnC[code][explanation_logger.language]

    instance = alloc.instance
    check_value(instance, algo_prefix="divide and choose: ")
    explanation_logger.info("\n" + _("algorithm_starts") + "\n")
    agents: list = list(instance.agents)
    if len(agents) != 3:
        raise ValueError("divide and choose is for 3 agents")

    def get_valuations(agent, items):
        # Helper function to get valuations of an agent for given items
        return {item: instance.agent_item_value(agent, item) for item in items}

    def give_bundle(agent, bundle):
        # for readability
        alloc.give_bundle(agent, bundle, logger=explanation_logger)

    def repartition(agent1, agent2, items):
        # Helper function for repartition for 2 agents
        agent_valuations: dict = get_valuations(agent2, items)
        explanation_logger.info("\n" + _("leximin_partition"), agent2, items)
        partition2 = leximin_partition(agent_valuations, n=2)
        explanation_logger.debug("\n" + _("partition_results"), 2, str(partition2))
        priorities_other = get_bundle_rankings(instance, agent1, partition2)

        explanation_logger.info("\n" + _("given_choice"), agent1)
        give_bundle(agent1, priorities_other[0])

        explanation_logger.info("\n" + _("remainder"), agent2)
        give_bundle(agent2, priorities_other[1])
        pass

    explanation_logger.info("1: " + _("leximin_partition"), agents[2], instance.items)
    agent3_valuations = get_valuations(agents[2], instance.items)
    partition1 = leximin_partition(agent3_valuations)
    explanation_logger.debug("\n" + _("partition_results"), 3, str(partition1))

    explanation_logger.info("\n" + _("calc_priorities"), agents[0], agents[1], agents=agents[:2])
    priorities1 = get_bundle_rankings(instance, agents[0], partition1)
    priorities2 = get_bundle_rankings(instance, agents[1], partition1)

    explanation_logger.info("\n" + _("first_priority_check"), agents[0], agents[1], agents=agents[:2])
    if priorities1[0] != priorities2[0]:
        explanation_logger.info("\n" + _("different_first_priority"), agents[0], agents[1], agents=agents[:2])
        give_bundle(agents[0], priorities1[0])
        give_bundle(agents[1], priorities2[0])
        explanation_logger.info("\n" + _("remainder"), agents[2], agents=agents[2])
        remainder = next(v for v in partition1 if v not in [priorities1[0], priorities2[0]])
        give_bundle(agents[2], remainder)
        return

    explanation_logger.info(
        "\n" + "Favorite bundles are the same. Checking second priorities for agents 1 and 2" + "\n")
    if priorities1[1] == priorities2[1]:
        explanation_logger.info("\n" + _("same_second_priority"), agents[1])
        explanation_logger.info("\n" + _("repartition"), agents[1])
        unified_bundle = priorities2[0] + priorities2[1]
        explanation_logger.info("\n" + _("remainder"), agents[2])
        give_bundle(agents[2], priorities2[2])

        repartition(agent1=agents[0], agent2=agents[1], items=unified_bundle)
        return

    explanation_logger.info("\n" + _("different_second_priority") + "\n")
    is_significant1 = is_significant_2nd_bundle(instance, agents[0], priorities1)
    is_significant2 = is_significant_2nd_bundle(instance, agents[1], priorities2)
    if is_significant1 and is_significant2:
        explanation_logger.info("\n" + _("has_significant_worth") + "\n")
        give_bundle(agents[0], priorities1[1])
        give_bundle(agents[1], priorities2[1])
        explanation_logger.info("\n" + _("remainder"), agents[2])
        give_bundle(agents[2], priorities1[0])
        return

    non_significant_agent = agents[1] if not is_significant2 else agents[0]
    other_agent = agents[1] if non_significant_agent == agents[0] else agents[0]

    explanation_logger.info("\n" + _("not_significant"), non_significant_agent)
    explanation_logger.info("\n" + _("repartition"), non_significant_agent)
    unified_bundle = priorities2[0] + priorities2[1] if not is_significant2 else priorities1[0] + priorities1[1]
    remainder = priorities2[2] if not is_significant2 else priorities1[2]
    explanation_logger.info("\n" + _("remainder"), agents[2])
    give_bundle(agents[2], remainder)

    repartition(agent1=other_agent, agent2=non_significant_agent, items=unified_bundle)
    # No need to return a value. The `divide` function returns the output.


def create_envy_graph(instance: Instance, allocation: dict):
    """
    >>> inst = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> alloc = {"Alice": [2], "Bob": [1], "Claire":[0]}
    >>> eg = create_envy_graph(inst, alloc)
    >>> expected = nx.DiGraph(
    ... [('Alice','Bob'), ('Alice','Claire'),    # Alice envies both Claire and Bob
    ... ('Bob','Alice'), ('Bob','Claire'),       # Bob envies both Alice and Claire
    ... ('Claire','Bob'), ('Claire','Alice')])  # Claire envies both Alice and Bob
    >>> nx.is_isomorphic(eg, expected)          # Equality check
    True
    """
    abvm = AgentBundleValueMatrix(instance, allocation, normalized=False)
    abvm.make_envy_matrix()
    mat: dict[str, dict[
        str, int]] = abvm.envy_matrix  # mat example {'Alice': {'Alice': 0, 'Bob': 11}, 'Bob': {'Alice': -11, 'Bob': 0}}
    envy_edges = [
        (agent, other)
        for agent, agent_status in mat.items()
        for other, envy in agent_status.items()
        if envy > 0
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(instance.agents)
    if envy_edges:
        graph.add_edges_from(envy_edges)
    return graph


def envy_reduction_procedure(alloc: dict[str, list], instance: Instance,
                             explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Procedure P for algo. 2: builds an envy graph from a given allocation, finds and reduces envy cycles.
    i.e. allocations with envy-cycles should and would be fixed here.

    :param alloc: the current allocation - updated within the procedure
    :param instance: the current instance
    :param explanation_logger: a logger
    :return: non envied agents

    Note: the example wouldn't provide envy cycle neccessarly,
    but it is easier to create envy than find an example with such.

    Example:
    >>> instance = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> allocator = {"Alice": [2],  # Alice envies both Claire and Bob
    ... "Bob": [1],                 # Bob envies both Alice and Claire
    ... "Claire":[0]}               # Claire envies both Alice and Bob

    >>> envy_reduction_procedure(allocator, instance)
    ['Alice', 'Bob', 'Claire']
    >>> allocator
    {'Alice': [1], 'Bob': [0], 'Claire': [2]}

    given an allocation a, we define its corresponding envy-graph g as follows.
 every agent is represented by a node in g and there is a directed edge from node i to node j iff
 vi(ai) < vi(aj), i.e., i is envies j. a directed cycle in g is called an envy-cycle. let c = i1 →
 i2 → ··· → it → i1 be such a cycle. if we reallocate aik+1 to agent ik for all k ∈ [t − 1], and
 reallocate ai1 to agent it, the number of edges of g will be strictly decreased. thus, by repeatedly
 using this procedure, we eventually get another allocation whose envy-graph is acyclic. it is shown
 in [lipton et al., 2004] that p can be done in time o(mn3).
    """
    # TODO: edit texts
    TEXTS = {
        "procedure_starts": {
            "he": "אלגוריתם 'שידוך מקסימום עם פיצוי' מתחיל:",
            "en": "envy reduction process starts",
        },
        "current_alloc": {
            "he": "סיבוב %d:",
            "en": "current allocation %s",
        },
        "define_envy_graph": {
            "he": "מספר המקומות הפנויים בכל קורס: %s",
            "en": "Defining envy graph",
        },
        "no_envy_edges": {
            "he": "לא קיבלת אף קורס, כי לא נשארו קורסים שסימנת שאת/ה מוכן/ה לקבל.",
            "en": "There is no envy among the agents\n procedure ends",
        },
        "no_envy_cycle": {
            "he": "הערך הגבוה ביותר שיכולת לקבל בסיבוב הנוכחי הוא %g. קיבלת את %s, ששווה עבורך %g.",
            "en": "There is no envy cycle within the envy-graph\n procedure ends",
        },
        "envy_cycle": {
            "he": "קיבלת את כל %d הקורסים שהיית צריך.",
            "en": "There is an envy cycle",
        },
        "reassignment": {
            "he": "קיבלת את כל %d הקורסים שהיית צריך.",
            "en": "Reassigning bundles for %s",
        },
        "final_alloc": {
            "he": "כפיצוי, הוספנו את ההפרש %g לקורס הכי טוב שנשאר לך, %s.",
            "en": "final allocation: %s.",
        },
    }

    def _(code: str):
        return TEXTS[code][explanation_logger.language]

    explanation_logger.info("\n" + _("procedure_starts"))
    explanation_logger.info("\n" + _("define_envy_graph"))
    envy_graph = create_envy_graph(instance, alloc)
    while not nx.is_directed_acyclic_graph(envy_graph):
        explanation_logger.info("\n" + _("envy_cycle"))
        envy_cycle = nx.find_cycle(envy_graph)
        explanation_logger.debug("envy cycle: %s", envy_cycle)
        new_alloc = {envious: alloc[envied] for envious, envied in envy_cycle}
        alloc.update(new_alloc)
        envy_graph = create_envy_graph(instance, alloc)
    explanation_logger.info("\n" + _("no_envy_cycle"))

    no_envy_agents = [node for node in envy_graph.nodes if envy_graph.in_degree(node) == 0]
    return no_envy_agents


def maximum_matching(instance, agents, items):
    """
    Computes a maximum weight matching M between agents and items, where the weight of edge
    between a ∈ agents and i ∈ items is given by the value of i according to a

    >>> inst = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> maximum_matching(inst, list(inst.agents), list(inst.items))
    [('Alice', 1), ('Bob', 0), ('Claire', 2)]
    >>> maximum_matching(inst, list(inst.agents), [3])
    [('Claire', 3)]
    """

    mat = [[instance.agent_item_value(agent, item) for item in items] for agent in agents]
    # mat = np.array([[10, 10, 6, 4, 2, 2, 2], [7, 5, 6, 6, 6, 2, 9], [2, 8, 8, 7, 5, 2, 3]])
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(mat, maximize=True)
    agent_ind = [agents[i] for i in row_ind]
    items_ind = [items[i] for i in col_ind]
    matching = list(zip(agent_ind, items_ind))

    return matching


def alloc_by_matching(alloc: AllocationBuilder):
    """
    Algorithm 2: Finds an 1/2mma or mmax allocation for any amount of agents and items,
    using graphs and weighted natchings.
    note: Only valuations are needed

    Examples:
    >>> divide(alloc_by_matching, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [1], 'Bob': [0], 'Claire': [2, 3]}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,8,8,7,5,2,3]}
    >>> divide(alloc_by_matching, valuations=val_7items)
    {'Alice': [0, 2, 5], 'Bob': [4, 6], 'Claire': [1, 3]}
    """
    # 1: Initiate L = N and R = M.
    instance = alloc.instance
    check_value(instance, algo_prefix="alloc by matching: ")
    agents = list(instance.agents)
    items = list(instance.items)
    alloc_dict = {agent: [] for agent in agents}
    # print("alloc_dict init %s", alloc_dict)
    # 2: Initiate Ai =∅ for all i∈N.
    # 3: while R =∅ do
    while items:
        # 4: Compute a maximum weight matching M between L and R, where the weight of edge
        # between i∈L and j∈R is given by vi(Ai∪{j})−vi(Ai).
        # If all edges have weight 0,then we compute a maximum cardinality matching M instead.
        matching = maximum_matching(instance, agents, items)
        # 5: For every edge(i, j)∈M, allocate j to i: Ai = Ai∪{j}
        # print("matching %s", matching)
        new_alloc = {a: bundle + [item]
                     for agent, bundle in alloc_dict.items()
                     for a, item in matching if a == agent}
        # print("new_alloc init %s", new_alloc)
        alloc_dict.update(new_alloc)
        # print("alloc_dict update %s", alloc_dict)
        # and exclude j from R: R = R\{j}.
        for _, item in matching:
            items.remove(item)

        agents = envy_reduction_procedure(alloc_dict, instance)
    # No need to return a value. The `divide` function returns the output.
    for agent, bundle in alloc_dict.items():
        alloc.give_bundle(agent, bundle)


if __name__ == "__main__":
    # import doctest
    #
    # print("\n", doctest.testmod(), "\n")
    # val_7items = {"Alice": [10, 10, 6, 4, 2, 2, 2], "Bob": [7, 5, 6, 6, 6, 2, 9], "Claire": [2, 8, 8, 7, 5, 2, 3]}
    # divide(alloc_by_matching, valuations=val_7items)
    import sys

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.DEBUG)

    # instance = Instance(valuations={"Alice": [10, 10, 6, 4], "Bob": [7, 5, 6, 6], "Claire": [2, 8, 8, 7]})
    # allocator = {"Alice": [2],  # Alice envies both Claire and Bob
    #             "Bob": [1],  # Bob envies both Alice and Claire
    #             "Claire": [0]}  # Claire envies both Alice and Bob
    # envy_reduction_procedure(allocator, instance, explanation_logger=ConsoleExplanationLogger())
    # print(allocator)
    divide(divide_and_choose_for_three, valuations={"Alice": [9, 10], "Bob": [7, 5], "Claire": [2, 8]},
           explanation_logger=ConsoleExplanationLogger())
    # divide(divide_and_choose_for_three, valuations={"Alice": [10, 10, 6, 4], "Bob": [7, 5, 6, 6], "Claire": [2, 8, 8, 7]}, explanation_logger=ConsoleExplanationLogger())
    # divide(divide_and_choose_for_three, valuations={"Alice": [2,4,6,7], "Bob": [5,7,3,5], "Claire":[2,2,2,2]}, explanation_logger=ConsoleExplanationLogger())
    # my_log = ConsoleExplanationLogger(language="he")
    # print(my_log.language)
    # inst = Instance(
    #     valuations={"Alice": [8, 5, 1, 5, 5, 3, 6, 9, 3, 3, 7, 5, 8, 8, 4, 10, 3, 8, 10, 2],
    #                 "Bob": [3, 5, 5, 3, 4, 9, 5, 5, 8, 1, 2, 6, 8, 6, 9, 1, 2, 8, 9, 7],
    #                 "Claire": [7, 8, 2, 9, 3, 2, 3, 8, 8, 8, 4, 10, 10, 6, 9, 10, 5, 3, 10, 3]})
    # alloc = divide(divide_and_choose_for_three, inst, explanation_logger=my_log)
    # print(alloc)
