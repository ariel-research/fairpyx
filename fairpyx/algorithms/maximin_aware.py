"""
An implementation of the algorithms in:
"Maximin-Aware Allocations of Indivisible Goods" by H. Chan, J. Chen, B. Li, and X. Wu (2019)
https://arxiv.org/abs/1905.09969
Programmer: Sonya Rybakov
Since: 2024-05

Disclaimer: all algorithms are on additive valuations
 """
import networkz as nx

from fairpyx import Instance, AllocationBuilder, divide, ExplanationLogger, AgentBundleValueMatrix


def leximin_partition(valuation, n: int = 3):
    """
     A leximin n-partition is a partition which divides M into n subsets and
    maximizes the lexicographical order when the values of the partitions are sorted in non-decreasing
    order. In other words, it maximizes the minimum value over all possible n partitions,
    and if there is a tie it selects the one maximizing the second minimum value, and so on
    >>> leximin_partition([9,10])
    [[1], [0], []]

    >>> leximin_partition([2,8,8,7])
    [[1], [2], [0, 3]]

    # leximin_partition([2,8,8,7,5,2,3]) # TODO: my partition could be wrong
    [[0, 5, 6], [1, 2], [3, 4]]

    >>> leximin_partition([2,2,2,2])
    [[0, 1], [2], [3]]
    """
    # Pair values with their indices
    indexed_values = list(enumerate(valuation))

    # Sort values in descending order along with their indices
    indexed_values.sort(key=lambda x: -x[1])

    # Initialize partitions and sums
    partitions = [[] for _ in range(n)]
    sums = [0] * n  # To track the sum of each partition

    for index, value in indexed_values:
        # Find the partition with the smallest sum
        min_index = sums.index(min(sums))
        partitions[min_index].append((index, value))
        sums[min_index] += value

    # Sort each partition by the indices to preserve the original order within partitions
    for part in partitions:
        part.sort()

    # Extract just the indices
    partition_indices = [[index for index, value in part] for part in partitions]

    return partition_indices


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
    {'Alice': [1,2], 'Bob': [0,5,6], 'Claire': [3,4]}

    step 3 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [0,3], 'Bob': [1], 'Claire': [2]}

    step 4-I allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,7,6,2], "Bob": [5,5,3,7], "Claire":[2,2,2,2]})
    {'Alice': [2], 'Bob': [3], 'Claire': [0,1]}

    step 4-II allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,7,6,4], "Bob": [5,5,3,7], "Claire":[2,2,2,2]})
    {'Alice': [0,1], 'Bob': [3], 'Claire': [2]}
    """

    TEXTS = {
        "algorithm_starts": {
            "he": "אלגוריתם 'שידוך מקסימום עם פיצוי' מתחיל:",
            "en": "Divide-and-Choose Algorithm for Three Agents starts",
        },
        "leximin_partition": {
            "he": "סיבוב %d:",
            "en": "Agent %s divides the the items:",
        },
        "remaining_seats": {
            "he": "מספר המקומות הפנויים בכל קורס: %s",
            "en": "Remaining seats in each course: %s",
        },
        "you_did_not_get_any": {
            "he": "לא קיבלת אף קורס, כי לא נשארו קורסים שסימנת שאת/ה מוכן/ה לקבל.",
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

    def _(code: str): return TEXTS[code][explanation_logger.language]
    instance = alloc.instance
    explanation_logger.info("\n" + _("algorithm_starts") + "\n")
    #  Agent 3 divides M as a leximin partition, which is denoted by A = (A1,A2,A3).
    explanation_logger.info("\n" + "Agent 3 divides the items as leximin partition" + "\n")
    agents = list(instance.agents)
    agent3_valuations = [instance.agent_item_value(agents[2], item) for item in instance.items]
    partition = leximin_partition(agent3_valuations)
    #  2. If agent 1’s and agent 2’s favorite bundles are different, each of them takes her favorite
    #  bundle and the remaining bundle is allocated to agent 3.
    explanation_logger.debug("\n" + "the leximin partition" + "\n")
    explanation_logger.info("\n" + "Calculating priorities for agent 1 and agent 2" + "\n")
    priorities1 = priorities2 = []
    for i, b in enumerate(partition):
        priorities1.append((i, instance.agent_bundle_value(agent=agents[0], bundle=b)))
        priorities2.append((i, instance.agent_bundle_value(agent=agents[1], bundle=b)))

    explanation_logger.info("\n" + "Checking second priorities for agent 1 and agent 2" + "\n")

    #  3. Otherwise (agent 1 and agent 2 have the same favorite bundle), we compare their second
    #  favorite bundle. If their second favorite bundles are also the same, agent 2 takes her favorite
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
    mat: dict = abvm.envy_matrix  # mat example {'Alice': {'Alice': 0, 'Bob': 11}, 'Bob': {'Alice': -11, 'Bob': 0}}
    envy_edges = []
    for agent in instance.agents:
        agent_status = mat[agent]
        for other, envy in agent_status.items():
            if envy > 0:
                envy_edges.append((agent, other))
    # envy_edges = [(agent, other) for agent in instance.agents for other, envy in mat[agent] if envy > 0]
    if not envy_edges:
        return None
    return nx.DiGraph(envy_edges)


def envy_reduction_procedure(alloc: AllocationBuilder,
                             explanation_logger: ExplanationLogger = ExplanationLogger()) -> AllocationBuilder:
    """
    Procedure P for algo. 2: builds an envy graph from a given allocation, finds and reduces envy cycles.
    i.e. allocations with envy-cycles should and would be fixed here.

    :param alloc: the current allocation
    :param explanation_logger: a logger
    :return: updated allocation with no envy cycles

    Note: the example wouldn't provide envy cycle neccessarly,
    but it is easier to create envy than find an example with such.

    Example:
    >>> instance = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> allocator = AllocationBuilder(instance)

    >>> allocator.give('Alice', 2) # Alice envies both Claire and Bob
    >>> allocator.give('Bob', 1) # Bob envies both Alice and Claire
    >>> allocator.give('Claire', 0) # Claire envies both Alice and Bob
    >>> reduced = envy_reduction_procedure(allocator).sorted()
    >>> reduced
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
    instance = alloc.instance
    curr_alloc = alloc.sorted()
    envy_graph = create_envy_graph(instance,curr_alloc)
    if envy_graph is None:
        return alloc
    cycles = sorted(nx.simple_cycles(envy_graph))
    envy_cycle = cycles
    if all(isinstance(item, list) for item in cycles):
        envy_cycle = max(cycles, key=len)  # the largest cycle
    relation = [(envy_cycle[i], envy_cycle[(i + 1) % len(envy_cycle)]) for i in range(len(envy_cycle))]
    new_alloc = AllocationBuilder(instance)
    for envious, envied in relation:
        new_alloc.give_bundle(envious, curr_alloc[envied])

    return new_alloc


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
