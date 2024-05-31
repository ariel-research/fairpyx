"""
An implementation of the algorithms in:
"Maximin-Aware Allocations of Indivisible Goods" by H. Chan, J. Chen, B. Li, and X. Wu (2019)
https://arxiv.org/abs/1905.09969
Programmer: Sonya Rybakov
Since: 2024-05

Disclaimer: all algorithms are on additive valuations
 """
import networkz as nx
from prtpy import partition, objectives as obj, outputtypes as out
from prtpy.partitioning import integer_programming

from fairpyx import Instance, AllocationBuilder, divide, ExplanationLogger, AgentBundleValueMatrix


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


def get_bundle_rankings(instance: Instance, agent, bundles: dict):
    """
    checks the ranking of bundles according to agent's preferences
    :param instance: the current instance
    :param agent: the questioned agent
    :param bundles: a dict which maps a bundle and its items
    :return: the dict bundle sorted according to the ranking, where the 1st key is the 1st priority
    >>> inst1 = Instance(valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    >>> get_bundle_rankings(instance=inst1, agent='Alice', bundles={'b1':[], 'b2':[0], 'b3':[1]})
    {'b3': [1], 'b2': [0], 'b1': []}

    >>> get_bundle_rankings(instance=inst1, agent='Bob', bundles={'b1':[], 'b2':[0], 'b3':[1]})
    {'b2': [0], 'b3': [1], 'b1': []}

    >>> inst2 = Instance(valuations={"Alice": [2,2,6,7], "Bob": [5,3,7,5], "Claire":[2,2,2,2]})
    >>> rank_a = get_bundle_rankings(instance=inst2, agent='Alice', bundles={'b1':[2], 'b2':[1], 'b3':[0, 3]})
    >>> rank_b = get_bundle_rankings(instance=inst2, agent='Bob', bundles={'b1':[2], 'b2':[1], 'b3':[0, 3]})
    >>> rank_a == rank_b    # instance that goes to step 4 has same ranking
    True
    """
    ranking = {k: v for k, v in sorted(bundles.items(),
                                       key=lambda item: instance.agent_bundle_value(agent, item[1]),
                                       reverse=True)}
    return ranking


def is_significant_2nd_bundle(instance: Instance, agent, bundles: dict) -> bool:
    """
    checks the significance of the second priority bundle of an agent

    :param instance: the current instance
    :param agent: the agent in question
    :param bundles: a dict which maps a bundle and its items, sorted by priorities
    :return: if the second bundle has significant value
    >>> inst2 = Instance(valuations={"Alice": [2,2,6,7], "Bob": [5,3,7,5], "Claire":[2,2,2,2]})
    >>> rank_a = get_bundle_rankings(instance=inst2, agent='Alice', bundles={'b1':[2], 'b2':[1], 'b3':[0, 3]})
    >>> rank_b = get_bundle_rankings(instance=inst2, agent='Bob', bundles={'b1':[2], 'b2':[1], 'b3':[0, 3]})

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
    {'Alice': [0], 'Bob': [1], 'Claire': []}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,8,8,7,5,2,3]}
    >>> divide(divide_and_choose_for_three, valuations=val_7items)
    {'Alice': [0,3,5], 'Bob': [2,6], 'Claire': [1 ,4]}

    step 3 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [0,3], 'Bob': [1], 'Claire': [2]}

    step 4-I allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,2,6,7], "Bob": [5,3,7,5], "Claire":[2,2,2,2]})
    {'Alice': [2], 'Bob': [3], 'Claire': [0,1]}

    step 4-II allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,4,6,7], "Bob": [5,3,7,5], "Claire":[2,2,2,2]})
    {'Alice': [0,1], 'Bob': [3], 'Claire': [2]}
    """

    TEXTS = {
        "algorithm_starts": {
            "he": "אלגוריתם חלק ובחר לשלושה סוכנים מתחיל",
            "en": "Divide-and-Choose Algorithm for Three Agents starts",
        },
        "leximin_partition": {
            "he": "מחלק את הפריטים %s סוכן ",
            "en": "Agent %s divides the the items:",
        },
        "partition_results": {
            "he": " %d-חלוקה: %s",
            "en": "%d-partition results: %s",
        },
        "you_did_not_get_any": {
            "he": "חישוב חשיבויות לסוכנים: %s, %s",
            "en": "You did not get any course, because no remaining course is acceptable to you.",
        },
        "your_course_this_iteration": {
            "he": "הערך הגבוה ביותר שיכולת לקבל בסיבוב הנוכחי הוא %g. קיבלת את %s, ששווה עבורך %g.",
            "en": "The maximum possible value you could get in this iteration is %g. You get course %s whose value for you is %g.",
        },
        "you_have_your_capacity": {
            "he": "קיבלת את כל %d הקורסים שהיית צריך.",
            "en": "You now have all %d courses that you needed.",
        },
        "as_compensation": {
            "he": "כפיצוי, הוספנו את ההפרש %g לקורס הכי טוב שנשאר לך, %s.",
            "en": "As compensation, we added the difference %g to your best remaining course, %s.",
        },
    }

    def _(code: str):
        return TEXTS[code][explanation_logger.language]

    instance = alloc.instance
    explanation_logger.info("\n" + _("algorithm_starts") + "\n")

    explanation_logger.info("\n" + "Agent 3 divides the items as leximin partition" + "\n")
    agents: list = list(instance.agents)
    agent3_valuations: dict = {item: instance.agent_item_value(agents[2], item) for item in instance.items}
    partition1: dict = {k: v for k, v in zip(range(3), leximin_partition(agent3_valuations))}
    explanation_logger.debug("\n" + "the first leximin partition" + "\n")
    explanation_logger.info("\n" + "Calculating priorities for agent 1 and agent 2" + "\n")
    priorities1 = get_bundle_rankings(instance, agents[0], partition1)
    priorities2 = get_bundle_rankings(instance, agents[1], partition1)
    explanation_logger.info("\n" + "Checking first priorities for agents 1 and 2" + "\n")
    if priorities1[0] != priorities2[0]:
        explanation_logger.info("\n" + "agent 1’s and agent 2’s favorite bundles are different, "
                                + "each of them takes her favorite bundle "
                                + "and the remaining bundle is allocated to agent 3" + "\n")
        alloc.give_bundle(agents[0], priorities1[0])
        alloc.give_bundle(agents[1], priorities2[0])
        remainder = [v for k, v in partition1 if (v != priorities1[0] and v != priorities2[0])]
        alloc.give_bundle(agents[2], remainder)
        return

    explanation_logger.info(
        "\n" + "Favorite bundles are the same. Checking second priorities for agents 1 and 2" + "\n")
    if priorities1[1] == priorities2[1]:
        explanation_logger.info("\n"
                                + "Second favorite bundles are the same. "
                                + "Agent 2 takes her 2 favorites for repartition "
                                + "and the remaining bundle is allocated to agent 3" + "\n")
        unified_bundle = priorities2[0] + priorities2[1]
        alloc.give_bundle(agents[2], priorities2[2])

        agent2_valuations: dict = {item: instance.agent_item_value(agents[1], item) for item in unified_bundle}
        partition2: dict = {k: v for k, v in zip(range(3), leximin_partition(agent2_valuations, n=2))}
        priorities1 = get_bundle_rankings(instance, agents[0], partition2)
        explanation_logger.info("\n"
                                + "Agent 1 takes her favorite from repartition"
                                + "and the remaining bundle is allocated to agent 2" + "\n")
        alloc.give_bundle(agents[0], priorities1[0])
        alloc.give_bundle(agents[1], priorities1[1])
        return
    else:
        explanation_logger.info("\n"
                                + "Second favorite bundles are not the same. "
                                + "Checking significant worth of second favorites" + "\n")
        is_significant1 = is_significant_2nd_bundle(instance, agents[0], priorities1)
        is_significant2 = is_significant_2nd_bundle(instance, agents[1], priorities2)
        if is_significant1 and is_significant2:
            explanation_logger.info("\n"
                                    + "There is significant worth of second favorites for both"
                                    + "Assigning second favorites accordingly, remainder goes to agent 3" + "\n")
            alloc.give_bundle(agents[0], priorities1[1])
            alloc.give_bundle(agents[1], priorities2[1])
            alloc.give_bundle(agents[2],  priorities1[0])
            return
        elif not is_significant2:
            explanation_logger.info("\n"
                                    + "There is significant worth of second favorites"
                                    + "Assigning second favorites accordingly, remainder goes to agent 3" + "\n")
            unified_bundle = priorities2[0] + priorities2[1]
            alloc.give_bundle(agents[2], priorities2[2])

            agent2_valuations: dict = {item: instance.agent_item_value(agents[1], item) for item in unified_bundle}
            partition2: dict = {k: v for k, v in zip(range(3), leximin_partition(agent2_valuations, n=2))}
            priorities1 = get_bundle_rankings(instance, agents[0], partition2)
            explanation_logger.info("\n"
                                    + "Agent 1 takes her favorite from repartition"
                                    + "and the remaining bundle is allocated to agent 2" + "\n")
            alloc.give_bundle(agents[0], priorities1[0])
            alloc.give_bundle(agents[1], priorities1[1])
            return
        else:
            unified_bundle = priorities1[0] + priorities1[1]
            alloc.give_bundle(agents[2], priorities1[2])

            agent1_valuations: dict = {item: instance.agent_item_value(agents[0], item) for item in unified_bundle}
            partition2: dict = {k: v for k, v in zip(range(3), leximin_partition(agent1_valuations, n=2))}
            priorities2 = get_bundle_rankings(instance, agents[1], partition2)
            explanation_logger.info("\n"
                                    + "Agent 2 takes her favorite from repartition"
                                    + "and the remaining bundle is allocated to agent 1" + "\n")
            alloc.give_bundle(agents[0], priorities2[0])
            alloc.give_bundle(agents[1], priorities2[1])
            return

    #  3. Otherwise (agent 1 and agent 2 have the same favorite bundle), we compare their second
    #  favorite bundle.
    #  If their second favorite bundles are also the same, agent 2 takes her favorite
    #  two bundles (w.o.l.g, say A1 and A2), and leaves A3 to agent 3. Next, agent 2 repartitions
    #  A1 ∪A2 by her leximin 2-partitions, denoted by B1 and B2. Agent 1 chooses her favorite
    #  one in B1 and B2, and the other one is allocated to agent 2.

    #  4. Finally, suppose their second favorite bundles are different. W.o.l.g, suppose v1(A1) ≥
    #  v1(A2) ≥ v1(A3) and v2(A1) ≥ v2(A3) ≥ v2(A2).
    #  (a) If 2 · v1(A2) > v1(A1) + v1(A3) and 2 · v2(A3) > v2(A1) + v2(A2), allocate A2 to
    #  agent 1, A3 to agent 2 and A1 to agent 3.
    #  (b) Otherwise (assume w.l.o.g. that 2 · v1(A2) ≤ v1(A1) + v1(A3)). Agent 2 takes A1
    #  and A3, which are her favorite two bundles, and leave A2 to agent 3. Next, agent 2
    #  repartitions A1 ∪ A3 by her leximin 2-partition C = (C1,C2), and let Agent 1 choose
    #  her favorite one in C1 and C2. The other one is allocated to agent 2.

    pass
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
    # envy_edges = [(agent, other) for agent in instance.agents for other, envy in mat[agent] if envy > 0]
    if not envy_edges:
        return nx.DiGraph()
    graph = nx.DiGraph()
    graph.add_nodes_from(instance.agents)
    graph.add_edges_from(envy_edges)
    return graph


def envy_reduction_procedure(alloc: dict[str, list], instance: Instance,
                             explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Procedure P for algo. 2: builds an envy graph from a given allocation, finds and reduces envy cycles.
    i.e. allocations with envy-cycles should and would be fixed here.

    :param alloc: the current allocation
    :param instance: the current instance
    :param explanation_logger: a logger
    :return: updated allocation with no envy cycles

    Note: the example wouldn't provide envy cycle neccessarly,
    but it is easier to create envy than find an example with such.

    Example:
    >>> instance = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> allocator = {"Alice": [2],  # Alice envies both Claire and Bob
    ... "Bob": [1],                 # Bob envies both Alice and Claire
    ... "Claire":[0]}               # Claire envies both Alice and Bob

    >>> envy_reduction_procedure(allocator, instance)
    >>> allocator
    {'Alice': [0], 'Bob': [2], 'Claire': [1]}

    given an allocation a, we define its corresponding envy-graph g as follows.
 every agent is represented by a node in g and there is a directed edge from node i to node j iff
 vi(ai) < vi(aj), i.e., i is envies j. a directed cycle in g is called an envy-cycle. let c = i1 →
 i2 → ··· → it → i1 be such a cycle. if we reallocate aik+1 to agent ik for all k ∈ [t − 1], and
 reallocate ai1 to agent it, the number of edges of g will be strictly decreased. thus, by repeatedly
 using this procedure, we eventually get another allocation whose envy-graph is acyclic. it is shown
 in [lipton et al., 2004] that p can be done in time o(mn3).
    """
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
            "en": "There is no envy cycle within the agents\n procedure ends",
        },
        "envy_cycle": {
            "he": "קיבלת את כל %d הקורסים שהיית צריך.",
            "en": "There is an envy cycle, begin reassignment",
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

    explanation_logger.info("\n" + _("procedure_starts") + "\n")
    envy_graph = create_envy_graph(instance, alloc)
    while not nx.is_directed_acyclic_graph(envy_graph):
        envy_cycle = nx.find_cycle(envy_graph)
        if len(envy_cycle) == 2:  # only two agents involved
            envious, envied = envy_cycle[0]
            alloc.update({envious: alloc[envied]})
        else:
            for envious, envied in envy_cycle:
                # if (envied, envious) not in seen:
                alloc.update({envious: alloc[envied]})
                # seen.add((envious, envied))
        envy_graph = create_envy_graph(instance, alloc)


def alloc_by_matching(alloc: AllocationBuilder):
    """
    Algorithm 2: Finds an 1/2mma or mmax allocation for any amount of agents and items,
    using graphs and weighted natchings.
    note: Only valuations are needed

    Examples:
    >>> divide(alloc_by_matching, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [0], 'Bob': [1], 'Claire': [2,3]}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,8,8,7,5,2,3]}
    >>> divide(alloc_by_matching, valuations=val_7items)
    {'Alice': [0,1], 'Bob': [4,5,6], 'Claire': [2,3]}
    """
    # 1: InitiateL = NandR = M.
    # 2: InitiateAi =∅foralli∈N.
    # 3: whileR =∅do
    # 4: ComputeamaximumweightmatchingMbetweenLandR, wheretheweightofedge
    # betweeni∈Landj∈Risgivenbyvi(Ai∪{j})−vi(Ai).Ifalledgeshaveweight0,
    # thenwecomputeamaximumcardinalitymatchingMinstead.
    # 5: Foreveryedge(i, j)∈M, allocatej
    # toi: Ai = Ai∪{j}
    # andexcludejfromR: R =
    # R\{j}.
    # 6: Aslongasthereisanenvy - cyclewithrespect
    # toA = (Ai)
    # i∈N, invokeprocedureP(tobe
    # describedlater).
    # 7: UpdateA = (Ai)
    # i∈NtobetheallocationsafterP.
    # 8: Updatethesetofagentsnotenviedbyanyotheragents:L = {i∈N:∀j∈N, vj(Aj)≥
    # vj(Ai)}.
    # 9: endwhile

    pass
    # No need to return a value. The `divide` function returns the output.
