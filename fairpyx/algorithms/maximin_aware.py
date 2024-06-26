"""
An implementation of the algorithms in:
"Maximin-Aware Allocations of Indivisible Goods" by H. Chan, J. Chen, B. Li, and X. Wu (2019)
https://arxiv.org/abs/1905.09969
Programmer: Sonya Rybakov
Since: 2024-05

Disclaimer: all algorithms are on additive valuations
 """
import logging
import sys

import networkz as nx
from prtpy import partition, objectives as obj, outputtypes as out
from prtpy.partitioning import integer_programming

from fairpyx import Instance, AllocationBuilder, divide, ExplanationLogger, AgentBundleValueMatrix

logger = logging.getLogger(__name__)


def divide_and_choose_for_three(alloc: AllocationBuilder, explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Algorithm 1: Finds an mma1 allocation for 3 agents using leximin-n-partition.
    note: Only valuations are needed

    Examples:
    step 2 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    {'Alice': [1], 'Bob': [0], 'Claire': []}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,9,8,7,5,2,3]}
    >>> divide(divide_and_choose_for_three, valuations=val_7items)
    {'Alice': [0, 1, 5], 'Bob': [2, 6], 'Claire': [3, 4]}

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
    TEXT = {
        "algorithm_starts": {
            "he": "אלגוריתם חלק ובחר לשלושה סוכנים מתחיל",
            "en": "Divide-and-Choose Algorithm for Three Agents starts",
        },
        "leximin_partition": {
            "he": "סוכן %s מחלק  את הפריטים  %s בחלוקת לקסימין ל %d חלקים",
            "en": "Agent %s makes a leximin partition of the items %s into %d parts",
        },
        "partition_results": {
            "he": "\t %d-חלוקה: %s",
            "en": "\t%d-partition results: %s",
        },
        "calc_priorities": {
            "he": "חישוב חשיבויות לסוכנים: %s, %s",
            "en": "Calculate priorities for the first two agents: %s, %s",
        },
        "priority": {
            "he": "סדר עדיפות: %s.",
            "en": "priorities: %s",
        },
        "different_first_priority": {
            "he": " עדיפות ראשונה שונה בין הסוכנים: %s, %s, כל אחד מקבל את העדיפות הראשונה שלו",
            "en": "agent %s’s and agent %s’s favorite bundles are different, each of them takes their favorite bundle ",
        },
        "same_first_priority": {
            "he": " עדיפות ראשונה שווה, בדיקת חשיבות שנייה לסוכנים: %s, %s",
            "en": "agent %s’s and agent %s’s favorite bundles are the same, checking second priorities",
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
            "he": "החבילה הנותרת הולכת לסוכן %s",
            "en": "the remaining bundle is allocated to %s",
        },
        "different_second_priority": {
            "he": " עדיפות שנייה שווה בין הסוכנים, בדיקת ערך משמעותי עבור עדיפות שנייה",
            "en": "Second favorite bundles are different, checking significant worth of second favorites",
        },
        "has_significant_worth": {
            "he": "יש ערך משמעותי לעדיפות השנייה עבור שני הסוכנים, העדיפות השנייה מוקצית בהתאם",
            "en": "There is significant worth of second favorites for both. Assigning second favorites accordingly",
        },
        "not_significant": {
            "he": "אין ערך משמעותי עבור הסוכן %s.",
            "en": "There is no significant worth of second favorites for agent %s. "
        },
        "final_allocation": {
            "he": "החבילה הנותרת מוקצית לסוכן %s.",
            "en": "the remaining bundle is allocated to %s ",
        },
        "repartition": {
            "he": "הסוכן %s לוקח את 2 החבילות בעדיפות עליונה לחלוקה מחדש.",
            "en": "Agent %s takes their 2 favorites for repartition",
        },
    }

    def _(code: str):
        return TEXT[code][explanation_logger.language]

    check_no_capacities(alloc.instance, algo_prefix="divide and choose: ") # check that capacites are not used.

    agents = list(alloc.remaining_agents())
    if len(agents) != 3:
        raise ValueError("This algorithm requires exactly three agents")

    def get_valuations(agent, items):
        # Helper function to get valuations of an agent for given items
        return {item: alloc.effective_value(agent, item) for item in items}

    def give_bundle(agent, bundle):
        # for readability
        alloc.give_bundle(agent, bundle, logger=explanation_logger)

    def repartition(agent1, agent2, items):
        # Helper function for repartition for 2 agents
        agent_valuations: dict = get_valuations(agent2, items)
        explanation_logger.info("\t" + _("leximin_partition"), agent2, items, 2)
        # agent 2 re-partitions items by their leximin 2-partitions
        partition2 = approx_leximin_partition(agent_valuations, n=2)
        explanation_logger.debug("\t" + _("partition_results"), 2, str(partition2))
        priorities_other = get_bundle_rankings(alloc.agent_bundle_value, agent1, partition2)
        # Agent 1 chooses her favorite one in the new partition,
        explanation_logger.info("\t" + _("given_choice"), agent1)
        give_bundle(agent1, priorities_other[0])
        # and the other one is allocated to agent 2
        explanation_logger.info("\n" + _("remainder"), agent2)
        give_bundle(agent2, priorities_other[1])
        pass

    explanation_logger.info("\n" + _("algorithm_starts"))
    # Step 1: Leximin partition based on the third agent's valuations
    explanation_logger.info("\n(1) " + _("leximin_partition"), agents[2], list(alloc.remaining_items()), 3)
    agent3_valuations = get_valuations(agents[2], alloc.remaining_items())
    partition1 = approx_leximin_partition(agent3_valuations)
    explanation_logger.debug(_("partition_results"), 3, str(partition1))

    # Step 2: Calculate priorities for the first two agents
    explanation_logger.info("\n(2) " + _("calc_priorities"), agents[0], agents[1], agents=agents[:2])
    priorities1 = get_bundle_rankings(alloc.agent_bundle_value, agents[0], partition1)
    priorities2 = get_bundle_rankings(alloc.agent_bundle_value, agents[1], partition1)
    explanation_logger.debug(_("priority"), priorities1, agents=agents[0])
    explanation_logger.debug(_("priority"), priorities2, agents=agents[1])

    # Check if the first priorities are different
    if priorities1[0] != priorities2[0]:
        # Each agent among the first two take their favorite bundle, the remaining bundle goes to third agent
        explanation_logger.info("\n(3) " + _("different_first_priority"), agents[0], agents[1], agents=agents[:2])
        give_bundle(agents[0], priorities1[0])
        give_bundle(agents[1], priorities2[0])
        explanation_logger.info(_("remainder"), agents[2], agents=agents[2])
        remainder = next(v for v in partition1 if v not in [priorities1[0], priorities2[0]])
        give_bundle(agents[2], remainder)
        return

    # Step 3: Favorite bundles are the same. Checking second priorities equivalence for agents 1 and 2
    explanation_logger.info("\n(3) " + _("same_first_priority"), agents[0], agents[1])
    if priorities1[1] == priorities2[1]:
        # agent 2 takes their favorite two bundles for re-partition,
        explanation_logger.info(_("same_second_priority"))
        explanation_logger.info(_("repartition"), agents[1])
        unified_bundle = priorities2[0] + priorities2[1]
        # and leaves the remainder to third agent
        explanation_logger.info(_("remainder"), agents[2])
        give_bundle(agents[2], priorities2[2])

        repartition(agent1=agents[0], agent2=agents[1], items=unified_bundle)
        return
    # Step 4: Second favorite bundles are different
    # checking significant worth of second bundle: 2 * #2-bundle > #1-bundle + #3-bundle
    explanation_logger.info("\n(4) " + _("different_second_priority"))
    is_significant1 = is_significant_2nd_bundle(alloc.agent_bundle_value, agents[0], priorities1)
    is_significant2 = is_significant_2nd_bundle(alloc.agent_bundle_value, agents[1], priorities2)
    if is_significant1 and is_significant2:
        # Step 4-a: Second favorite bundles have significant worth for both
        #  Each agent among the first two take their second favorite bundle, the remaining bundle goes to third agent
        explanation_logger.info("\t a)" + _("has_significant_worth"))
        give_bundle(agents[0], priorities1[1])
        give_bundle(agents[1], priorities2[1])
        explanation_logger.info("\t" + _("remainder"), agents[2])
        give_bundle(agents[2], priorities1[0])
        return

    # Step 4-b: At least one of the agents doesn't have a significant worth
    # non significant agent - the agent whose #2-bundle isn't significantly worthy
    non_significant_agent = agents[1] if not is_significant2 else agents[0]
    other_agent = agents[1] if non_significant_agent == agents[0] else agents[0]
    explanation_logger.info("\t b) " + _("not_significant"), non_significant_agent)
    explanation_logger.info("\t" + _("repartition"), non_significant_agent)
    # non significant agent takes their favorite two bundles for re-partition
    unified_bundle = priorities2[0] + priorities2[1] if not is_significant2 else priorities1[0] + priorities1[1]
    # and leaves the remainder to the third agent
    remainder = priorities2[2] if not is_significant2 else priorities1[2]
    explanation_logger.info("\t" + _("remainder"), agents[2])
    give_bundle(agents[2], remainder)

    repartition(agent1=other_agent, agent2=non_significant_agent, items=unified_bundle)


def alloc_by_matching(alloc: AllocationBuilder, explanation_logger=ExplanationLogger()):
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
    >>> val_envycycle = {"Alice": [14, 17, 2], "Bob": [14, 19, 6]}
    >>> divide(alloc_by_matching, valuations=val_envycycle)
    {'Alice': [1], 'Bob': [0, 2]}
    >>> val_envycycle_2 = {"Alice": [57, 34, 22, 19, 14, 6], "Bob": [59, 34, 26, 17, 14, 2]}
    >>> divide(alloc_by_matching, valuations=val_envycycle_2)
    {'Alice': [0, 3], 'Bob': [1, 2, 4, 5]}
    """
    TEXTS = {
        "algorithm_starts": {
            "he": "הליך צמצום קנאה מתחיל",
            "en": "Algorithm allocation by matching starts.",
        },
        "initialization": {
            "he": "איתחול R:= רשימת פריטים, L:=רשימת סוכנים, A:=מיפוי הקצאות לכל סוכן (ריק)",
            "en": "initializing R:=items L:=agents A:=allocation (empty for each agent)",
        },
        "curr_agents": {
            "he": "רשימת הסוכנים כרגע: %s",
            "en": "\tCurrent agents: %s",
        },
        "curr_items": {
            "he": "רשימת הפריטים כרגע: %s",
            "en": "\tCurrent items: %s",
        },
        "compute_matching": {
            "he": "חשב שידוך מקסימלי בין סוכנים לפריטים, כאשר כל צלע בין סוכן לפריט משקלו כערך הפריט לפי הסוכן",
            "en": "Compute a maximum weight matching M between items and agents, "
                  + "where the weight of edge between i∈agents and j∈items is given by valuation of agent i on item j"
        },
        "alloc_by_matching": {
            "he": "עבור כל צלע (פריט, סוכן) בשידוך, הקצה את הפריט לסוכן",
            "en": "For every edge (i, j) ∈ M, add item j to agent i's bundle, ",
        },
        "exclude_items": {
            "he": "והוצא את הפריט מרשימת הפריטים",
            "en": "and remove j from items.",
        },
        "new_alloc": {
            "he": "הקצאה החדשה %s",
            "en": "new allocation: %s.",
        },
        "updated_alloc": {
            "he": "הקצאה מעודכנת %s",
            "en": "updated allocation: %s.",
        },
        "matching": {
            "he": "תוצאת השידוך %s",
            "en": "matching: %s.",
        },

    }

    def _(code: str):
        return TEXTS[code][explanation_logger.language]

    instance = alloc.instance
    check_no_capacities(instance, algo_prefix="alloc by matching: ")

    # 1: Initiate L = N and R = M.
    explanation_logger.info(_("algorithm_starts"))
    explanation_logger.info("\n" + _("initialization"))
    agents = list(alloc.remaining_agents())  # L = N := group of agents
    items = list(alloc.remaining_items())  # R = M := list of items to allocate

    # 2: Initiate A_i =∅ for all i ∈ N.
    # NOTE: the allocation is handled in a dict, to allow exchange along envy cycles.
    alloc_dict = {agent: [] for agent in agents}

    # 3: while R =∅ do
    explanation_logger.debug("\t" + _("curr_agents"), agents)
    explanation_logger.debug("\t" + _("curr_items"), items)

    while items:
        # 4: Compute a maximum weight matching M between L and R, where the weight of edge
        # between i∈L and j∈R is given by vi(Ai∪{j})−vi(Ai).
        # If all edges have weight 0,then we compute a maximum cardinality matching M instead.
        explanation_logger.info("\n" + _("compute_matching"))
        matching = maximum_matching(instance, agents, items)
        explanation_logger.debug("\t" + _("matching"), str(matching))

        # 5: For every edge(i, j)∈M, allocate j to i: Ai = Ai∪{j}
        explanation_logger.info(_("alloc_by_matching"))
        new_alloc = {a: bundle + [item]
                     for agent, bundle in alloc_dict.items()
                     for a, item in matching if a == agent}
        # explanation_logger.debug(_("new_alloc"), new_alloc)
        alloc_dict.update(new_alloc)
        explanation_logger.debug(_("updated_alloc"), alloc_dict)
        # and exclude j from R: R = R\{j}.
        # explanation_logger.info(_("exclude_items"))
        to_remove = [item for _, item in matching]
        items = [item for item in items if item not in to_remove]
        explanation_logger.debug(_("curr_items"), items)
        non_envied_agents = envy_reduction_procedure(alloc_dict, instance, explanation_logger=explanation_logger)
        agents = non_envied_agents
        explanation_logger.debug(_("curr_agents"), agents)

    for agent, bundle in alloc_dict.items():
        alloc.give_bundle(agent, bundle, logger=explanation_logger)


# ----------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------
def check_no_capacities(instance: Instance, algo_prefix: str):
    for item in instance.items:
        if instance.item_capacity(item) != 1:
            raise ValueError(algo_prefix + "item capacity restricted to only one")

    for agent in instance.agents:
        if instance.agent_capacity(agent) < instance.num_of_items:
            raise ValueError(algo_prefix + "agent capacity should be as many as the items")


# ----------------------------------------------------------
# Helper Functions - Divide ans Choose
# ----------------------------------------------------------
def approx_leximin_partition(valuation: dict, n: int = 3, result: out.OutputType = out.Partition):
    """
    Provides approximate leximin partition

    **Info:**
    A leximin n-partition is a partition which divides the items into n subsets and
    maximizes the lexicographical order when the values of the partitions are sorted in non-decreasing
    order. In other words, it maximizes the minimum value over all possible n partitions,
    and if there is a tie it selects the one maximizing the second minimum value, and so on.

    Since such partition is NP-Hard, a partition where it maximises the smallest sum is used instead.

    :param valuation: a dictionary to represent the items and their valuations
    :param n: the number of subsets required
    :param result: the desired result, default is a partition represented by item keys, anything else is for testing and
     debugging purposes


    >>> print(approx_leximin_partition({0:9,1:10}))
    [[], [0], [1]]

    >>> approx_leximin_partition({0:2,1:8,2:8,3:7},result=out.PartitionAndSumsTuple)
    (array([8., 8., 9.]), [[1], [2], [0, 3]])

    >>> sorted(approx_leximin_partition({0:2,1:9,2:8,3:7,4:5,5:2,6:3}))
    [[0, 2, 5], [1, 6], [3, 4]]

    >>> approx_leximin_partition({0:2,1:2,2:2,3:2},result=out.PartitionAndSumsTuple)
    (array([2., 2., 4.]), [[2], [1], [0, 3]])
    """
    prt = partition(algorithm=integer_programming.optimal, numbins=n, items=valuation, outputtype=result,
                    objective=obj.MaximizeSmallestSum)
    return prt


def get_bundle_rankings(agent_bundle_value:callable, agent, bundles: list) -> list:
    """
    checks the ranking of bundles according to agent's preferences
    :param instance: the current instance
    :param agent: the questioned agent
    :param bundles: a list of bundles
    :return: the bundle list  sorted according to the ranking
    >>> inst1 = Instance(valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    >>> get_bundle_rankings(inst1.agent_bundle_value, agent='Alice', bundles=[[], [0], [1]])
    [[1], [0], []]

    >>> get_bundle_rankings(inst1.agent_bundle_value, agent='Bob', bundles=[[], [0], [1]])
    [[0], [1], []]

    >>> inst2 = Instance(valuations={"Alice": [2,2,6,7], "Bob": [5,3,7,5], "Claire":[2,2,2,2]})
    >>> rank_a = get_bundle_rankings(inst2.agent_bundle_value, agent='Alice', bundles=[[2], [1], [0, 3]])
    >>> rank_b = get_bundle_rankings(inst2.agent_bundle_value, agent='Bob', bundles=[[2], [1], [0, 3]])
    >>> rank_a == rank_b    # instance that goes to step 4 has same ranking
    True
    """
    ranking = sorted(bundles, key=lambda bundle: agent_bundle_value(agent, bundle), reverse=True)
    return ranking


def is_significant_2nd_bundle(agent_bundle_value:callable, agent, bundles: list) -> bool:
    """
    checks the significance of the second priority bundle of an agent

    :param instance: the current instance
    :param agent: the agent in question
    :param bundles: a dict which maps a bundle and its items, sorted by priorities
    :return: if the second bundle has significant value

    >>> inst2 = Instance(valuations={"Alice": [2,2,6,7], "Bob": [5,7,3,5], "Claire":[2,2,2,2]})
    >>> is_significant_2nd_bundle(inst2.agent_bundle_value, agent='Alice', bundles=[[0, 3], [2], [1]])
    True
    >>> is_significant_2nd_bundle(inst2.agent_bundle_value, agent='Bob', bundles=[[0, 3], [1], [2]])
    True
    """
    return (agent_bundle_value(agent, bundles[1]) * 2) > agent_bundle_value(agent, bundles[0]) \
           + agent_bundle_value(agent, bundles[2])


# ----------------------------------------------------------
# Helper Functions - Alloc by Matching
# ----------------------------------------------------------
def create_envy_graph(instance: Instance, allocation: dict,
                      explanation_logger: ExplanationLogger = ExplanationLogger()):
    """
    Creates an envy-graph G

    **Info:**
    Every agent is represented by a node in G and there is a directed edge from node i to node j iff i envies j.

    :param instance: the current instance
    :param allocation: the current allocation which G is based on
    :return: Envy-Graph G

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
    TEXTS = {
        "envy_edges": {
            "he": "צלעות קנאה %s",
            "en": "envy edges %s",
        },
    }

    def _(code: str):
        return TEXTS[code][explanation_logger.language]

    abvm = AgentBundleValueMatrix(instance, allocation, normalized=False)
    # abvm valuation example:
    # valuations = {"Alice": {"c1": 11, "c2": 22}, "Bob": {"c1": 33, "c2": 44}}
    # allocation = {"Alice": ["c1"], "Bob": ["c2"]}
    # {'Alice': {'Alice': 11, 'Bob': 22}, 'Bob': {'Alice': 33, 'Bob': 44}}
    abvm.make_envy_matrix()
    mat: dict[str, dict[str, int]] = abvm.envy_matrix
    # mat example: {'Alice': {'Alice': 0, 'Bob': 11}, 'Bob': {'Alice': -11, 'Bob': 0}}
    # Alice envies Bob
    envy_edges = [
        (agent, other)
        for agent, agent_status in mat.items()
        for other, envy in agent_status.items()
        if envy > 0
    ]
    explanation_logger.debug("\t" + _("envy_edges"), envy_edges)
    graph = nx.DiGraph()
    graph.add_nodes_from(instance.agents)
    if envy_edges:
        graph.add_edges_from(envy_edges)
    return graph


def envy_reduction_procedure(alloc: dict[str, list], instance: Instance,
                             explanation_logger: ExplanationLogger = ExplanationLogger())->list:
    """
    Procedure P for algo. 2: builds an envy graph from a given allocation, finds and reduces envy cycles.
    i.e. allocations with envy-cycles should and would be fixed here.

    **Info:**
    Given an allocation A, we define its corresponding envy-graph G as follows.
    Every agent is represented by a node in g and there is a directed edge from node i to node j iff i is envies j.
    A directed cycle in g is called an envy-cycle.
    let c = i_1 → i_2 → ··· → i_t → i_1 be such a cycle.
    if we reallocate A_i_k+1 to agent i_k for all k ∈ [t − 1], and reallocate ai1 to agent i_t,
    the number of edges of G will be strictly decreased.
    Thus, by repeatedly using this procedure, we eventually get another allocation whose envy-graph is acyclic.

    :param alloc: the current allocation - updated within the procedure
    :param instance: the current instance
    :param explanation_logger: a logger
    :return: non envied agents

    Note: the example wouldn't provide envy cycle neccessarly,
    but it is easier to create envy than find an example with such.

    **Example:**
    >>> instance = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> allocator = {"Alice": [2],  # Alice envies both Claire and Bob
    ... "Bob": [1],                 # Bob envies both Alice and Claire
    ... "Claire":[0]}               # Claire envies both Alice and Bob

    >>> envy_reduction_procedure(allocator, instance)
    ['Alice', 'Bob', 'Claire']
    >>> allocator
    {'Alice': [1], 'Bob': [0], 'Claire': [2]}
    """
    TEXTS = {
        "procedure_starts": {
            "he": "הליך צמצום קנאה מתחיל",
            "en": "Envy-cycle reduction starts",
        },
        "current_alloc": {
            "he": "הקצאה נוכחית %s",
            "en": "Current allocation %s",
        },
        "define_envy_graph": {
            "he": "מגדיר גרף קנאה",
            "en": "Defining envy graph",
        },
        "no_envy_cycle": {
            "he": "אין מעגל קנאה בדרף הקנאה, ההליך מסתיים",
            "en": "There is no envy cycle within the envy-graph. Procedure ends",
        },
        "envy_cycle": {
            "he": "יש מעגל קנאה, מקצה מחדש חבילות",
            "en": "There is an envy cycle - Reassigning bundles",
        },
        "reassignment_for": {
            "he": "הקצאה מחדש עבור %s",
            "en": "Reassignment for %s.\n",
        },
        "new_alloc": {
            "he": "הקצאה החדשה",
            "en": "New allocation: %s.\n",
        },
    }

    def _(code: str):
        return TEXTS[code][explanation_logger.language]

    explanation_logger.info("\n" + _("procedure_starts"))
    explanation_logger.debug("\t" + _("current_alloc"), alloc)
    explanation_logger.info(_("define_envy_graph"))
    envy_graph = create_envy_graph(instance, alloc, explanation_logger)
    while not nx.is_directed_acyclic_graph(envy_graph):
        envy_cycle = nx.find_cycle(envy_graph)
        explanation_logger.info(_("envy_cycle"))
        explanation_logger.debug("\t" + _("reassignment_for"), envy_cycle)
        new_alloc = {envious: alloc[envied] for envious, envied in envy_cycle}
        explanation_logger.debug("\t" + _("new_alloc"), new_alloc)
        alloc.update(new_alloc)
        envy_graph = create_envy_graph(instance, alloc, explanation_logger)
    explanation_logger.info(_("no_envy_cycle"))
    # return non envied agents, i.e. those who have no inward edges
    non_envied_agents = [node for node in envy_graph.nodes if envy_graph.in_degree(node) == 0]
    return non_envied_agents


def maximum_matching(instance: Instance, agents: list, items: list):
    """
    Computes an assignment between agents and items.

    **Info:**
    A matching between agents and items where the weight of edge between
    i ∈ agents and j ∈ items is given by the value of item j according to agent i

    :param instance: the current instance - for agent item value
    :param agents: list of agents - which we attempt to fully match
    :param items: list of items

    >>> inst = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> maximum_matching(inst, list(inst.agents), list(inst.items))
    [('Alice', 1), ('Bob', 0), ('Claire', 2)]
    >>> maximum_matching(inst, list(inst.agents), [3])
    [('Claire', 3)]
    """

    mat = [[instance.agent_item_value(agent, item) for item in items] for agent in agents]

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(mat, maximize=True)
    agent_ind = [agents[i] for i in row_ind]
    items_ind = [items[i] for i in col_ind]
    matching = list(zip(agent_ind, items_ind))

    return matching


if __name__ == "__main__":
    import doctest
    print("\n", doctest.testmod(), "\n")
    # sys.exit()

    from fairpyx import ConsoleExplanationLogger
    from fairpyx.adaptors import divide_random_instance
    console_explanation_logger = ConsoleExplanationLogger()

    # inst = Instance(
    #     valuations={"Alice": [8, 5, 1, 5, 5, 3, 6, 9, 3, 3, 7, 5, 8, 8, 4, 10, 3, 8, 10, 2],
    #                 "Bob": [3, 5, 5, 3, 4, 9, 5, 5, 8, 1, 2, 6, 8, 6, 9, 1, 2, 8, 9, 7],
    #                 "Claire": [7, 1, 2, 9, 3, 2, 3, 8, 8, 7, 4, 10, 10, 6, 9, 10, 5, 3, 10, 3]})
    # alloc = divide(divide_and_choose_for_three, inst, explanation_logger=console_explanation_logger)

    # num_of_agents = 3
    # num_of_items = 19
    # divide_random_instance(algorithm=divide_and_choose_for_three, 
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, 
    #                        agent_capacity_bounds=[num_of_items,num_of_items], item_capacity_bounds=[1,1], 
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=1000,
    #                        explanation_logger=console_explanation_logger,
    #                        random_seed=2)

    # inst = Instance(
    #     valuations={"Alice": [10, 10, 6, 4], 
    #                 "Bob": [7, 5, 6, 6], 
    #                 "Claire": [2, 8, 8, 7]})
    # alloc = divide(alloc_by_matching, inst, explanation_logger=console_explanation_logger)

    inst = Instance(  # exemplifies an envy-cycle
        valuations={"Alice": [57, 34, 22, 19, 14, 6], 
                    "Bob": [59, 34, 26, 17, 14, 2]})
    alloc = divide(alloc_by_matching, inst, explanation_logger=console_explanation_logger)


    # num_of_agents = 3
    # num_of_items = 19
    # divide_random_instance(algorithm=alloc_by_matching, 
    #                        num_of_agents=num_of_agents, num_of_items=num_of_items, 
    #                        agent_capacity_bounds=[num_of_items,num_of_items], item_capacity_bounds=[1,1], 
    #                        item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.5,1.5], normalized_sum_of_values=1000,
    #                        explanation_logger=console_explanation_logger,
    #                        random_seed=2)
