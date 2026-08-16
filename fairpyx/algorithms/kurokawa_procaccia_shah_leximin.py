"""
An implementation of the algorithm in:
"Leximin Allocations in the Real World", by D. Kurokawa, A. D. Procaccia, and N. Shah (2018), https://doi.org/10.1145/3274641

Programmer: Lior Trachtman
Date: 2025-05-05
"""

import logging
from fairpyx.allocations import AllocationBuilder
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpContinuous, value, PULP_CBC_CMD
from itertools import product

# Setup basic logging configuration (you can customize format and level)
logger = logging.getLogger(__name__)

def feasibility_ilp(S, items, demands, capacities, preferences):
    """
    Solve the FeasibilityILP subproblem for a given subset of agents S.

    Args:
        S (set[int]): Subset of agents to include in the ILP. For example, {1, 2, 3}.
        items (set[str]): Set of all available items or facilities (e.g., {'a', 'b'}).
        demands (dict[int, int]): Dictionary mapping each agent to their demand (e.g., {1: 1, 2: 2}).
        capacities (dict[str, int]): Dictionary mapping each item to its total capacity (e.g., {'a': 2}).
        preferences (dict[int, set[str]]): Dictionary mapping each agent to the set of items they value
            (i.e., items they are willing to receive).

    Returns:
        list[tuple[int, str]] | None:
            A list of (agent, item) pairs representing a deterministic assignment
            if a feasible solution exists. Otherwise, returns None if the problem is infeasible.

    This function solves the following ILP (from the LeximinPrimal algorithm):

        ∑_{f ∈ Fi} y_{i,f} ≥ 1           for all agents i ∈ S  (each agent gets at least one preferred item)
        ∑_{i ∈ S : f ∈ Fi} d_i·y_{i,f} ≤ c_f  for all items f ∈ items  (do not exceed item capacities)
        y_{i,f} ∈ {0, 1}                 for all i ∈ S and f ∈ Fi (binary decision variables)

    Doctest examples:
    >>> # Feasible case: agents 1 and 2 have non-overlapping preferences
    >>> S = {1, 2}
    >>> items = {'a', 'b'}
    >>> demands = {1: 1, 2: 1}
    >>> capacities = {'a': 1, 'b': 1}
    >>> preferences = {1: {'a'}, 2: {'b'}}
    >>> result = feasibility_ilp(S, items, demands, capacities, preferences)
    >>> sorted(result)
    [(1, 'a'), (2, 'b')]

    >>> # Infeasible case: both agents want 'a', but only one unit is available
    >>> S = {1, 2}
    >>> items = {'a'}
    >>> demands = {1: 1, 2: 1}
    >>> capacities = {'a': 1}
    >>> preferences = {1: {'a'}, 2: {'a'}}
    >>> feasibility_ilp(S, items, demands, capacities, preferences) is None
    True
    """

    logger.info("\n=== Starting FeasibilityILP ===")
    logger.info(f"Agents subset: {S}")

    # === Variables: y_{i,f} ∈ {0,1} for i ∈ S, f ∈ Fi (preferences[i]) ===
    prob = LpProblem("FeasibilityILP", LpMaximize)

    y = {
        (i, f): LpVariable(f"y_{i}_{f}", cat=LpBinary)
        for i in S
        for f in preferences[i]
    }
    logger.debug(f"Created {len(y)} binary variables for agent-facility assignments")

    # === Constraint 1: ∑_{f ∈ Fi} y_{i,f} ≥ 1 for all i ∈ S ===
    # Each agent must receive at least one preferred facility
    for i in S:
        prob += lpSum(y[i, f] for f in preferences[i]) >= 1
        logger.debug(f"Passed algorithm constraint: agent {i} assigned at least one preferred facility")

    # === Constraint 2: ∑_{i ∈ S : f ∈ Fi} d_i * y_{i,f} ≤ c_f for all f ∈ M ===
    # Total assigned demand for each facility must not exceed its capacity
    for f in items:
        prob += lpSum(
            demands[i] * y[i, f]
            for i in S
            if f in preferences[i]
        ) <= capacities[f]
        logger.debug(f"Passed algorithm constraint for facility '{f}': sum demand ≤ {capacities[f]}")

    # === Solve the ILP ===
    prob.solve(PULP_CBC_CMD(msg=0))

    # === If feasible, return the assignment as a list of (i, f) pairs ===
    if prob.status == 1:
        assigned = [(i, f) for (i, f), var in y.items() if var.varValue == 1]
        logger.info(f"FeasibilityILP found feasible assignment: {assigned}")
        return assigned

    # === If no feasible solution found, return None ===
    logger.info("FeasibilityILP found no feasible assignment")
    return None



def primal_lp(feasible_sets, R, agents, p_star):
    """
    Arguments:
        feasible_sets (list of (set, list)):
            Each element is (agents_in_S, allocation), where allocation is a list of (agent, item) pairs.

        R (set of int):
            Agents whose allocation probability must be ≥ M.

        agents (list of int):
            All agents in the problem instance.

        p_star (dict of int → float):
            Fixed probabilities for agents not in R.

    Solve the PrimalLP step of LeximinPrimal:
    Maximize M subject to:
        - pi ≥ M       for i ∈ R
        - pi = p*_i    for i ∈ N without R
        - pi = sum_{S: i ∈ S} xS
        - sum xS = 1
        - xS ≥ 0
    Returns:
        M  → current Leximin value (minimum guaranteed probability)
        pi → dictionary of probabilities per agent i ∈ R
        xS → LP variables representing distribution over feasible allocations

    Doctest example:
    >>> feasible_sets = [
    ...     ({1, 2}, [(1, 'a'), (2, 'b')]),
    ...     ({1}, [(1, 'a')])
    ... ]
    >>> R = {1, 2}
    >>> agents = [1, 2]
    >>> p_star = {1: 0.0, 2: 0.0}
    >>> M, pi, xS = primal_lp(feasible_sets, R, agents, p_star)
    >>> isinstance(M, float)
    True
    >>> sorted(pi.keys()) == sorted(R)
    True
    >>> all(0.0 <= val <= 1.0 for val in pi.values())
    True
    >>> all(isinstance(var, type(xS[next(iter(xS))])) for var in xS.values())
    True
    """
    logger.info("\n=== Starting PrimalLP ===")

    # === Objective: Maximize M ===
    prob = LpProblem("PrimalLP", LpMaximize)
    M = LpVariable("M", lowBound=0)  # Leximin value to maximize
    logger.debug("Created variable M for leximin value")

    # === Variables: xS ≥ 0 for each feasible allocation S ∈ S ===
    xS = {
        tuple(alloc): LpVariable(f"x_{idx}", lowBound=0, cat=LpContinuous)
        for idx, (_, alloc) in enumerate(feasible_sets)
    }
    logger.debug(f"Created {len(xS)} variables xS for feasible allocations")

    # === Constraint: ∑ xS = 1 → total probability mass must be 1 ===
    prob += lpSum(xS.values()) == 1
    logger.debug("Passed algorithm constraint: sum of xS variables equals 1")

    # === Constraints for each agent i ∈ N ===
    for i in agents:
        # pi = sum of xS for all allocations where agent i appears
        pi_expr = lpSum(xS[alloc] for alloc in xS if any(agent_id == i for (agent_id, _) in alloc))
        if i in R:
            prob += pi_expr >= M
            logger.debug(f"Passed algorithm constraint for agent {i}: pi >= M")
        else:
            prob += pi_expr == p_star[i]
            logger.debug(f"Passed algorithm constraint for agent {i}: pi == p_star[{i}] = {p_star[i]}")

    # === Final objective: maximize M - ε * sum(xS) for strict complementarity ===
    # This slight perturbation forces the solver to prefer solutions where fewer xS variables are non-zero,
    # making the solution strictly complementary
    # Add agent-wise pi[i] variables
    pi_vars = {i: LpVariable(f"pi_{i}", lowBound=0) for i in agents}

    # Replace pi_expr with constraint linking xS to pi[i]
    for i in agents:
        prob += pi_vars[i] == lpSum(xS[alloc] for alloc in xS if any(agent_id == i for (agent_id, _) in alloc))

    # Leximin constraints
    for i in R:
        prob += pi_vars[i] >= M
    for i in set(agents) - set(R):
        prob += pi_vars[i] == p_star[i]

    # find prob with maximized M
    epsilon = 1e-5
    prob += M + epsilon * lpSum(len(alloc) * xS[alloc] for alloc in xS)

    # === Solve LP ===
    from pulp import PULP_CBC_CMD

    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        logger.error(f"Primal LP is infeasible, status code: {prob.status}")
        raise Exception("Primal LP is infeasible")

    # === Detailed reporting of solution ===
    logger.info("\n=== LP Solution: Allocation Weights (xS) ===")
    for S in xS:
        val = value(xS[S])
        if val is not None and val > 1e-6:
            logger.info(f"Allocation {S} -> Weight: {val:.6f}")
        else:
            logger.debug(f"Allocation {S} -> Weight: {val:.6f}")

    # === Compute actual values of pi for i ∈ R ===
    pi = {}
    for i in R:
        pi_val = sum(value(xS[S]) for S in xS if any(s[0] == i for s in S))
        pi[i] = pi_val
        logger.debug(f"Inserting pi[{i}] = {pi_val:.6f} its weight")

    M_val = value(M)
    logger.info(f"\nPrimalLP solved successfully: M = {M_val:.6f}")

    return M_val, pi, xS

from itertools import combinations, product
from collections import Counter
from functools import lru_cache

@lru_cache(maxsize=None)
def generate_agent_bundles(preference_tuple, capacity):
    """
    Generates all bundles for an agent:
    - Agent can only receive multiple copies of ONE item
    - Total quantity ≤ capacity
    """
    return [
        (item,) * qty
        for item in preference_tuple
        for qty in range(1, capacity + 1)
    ]

def generate_feasible_sets(agents, items, demands, capacities, preferences):
    """
    Efficient generator for all feasible deterministic allocations over agent subsets.

    Each agent can receive one item type only (multiple copies),
    and items must be exclusively assigned (each item to at most one agent).

    Yields:
        (frozenset, list of (agent, item)) for each valid allocation
    """
    logger.info("Enumerating all feasible allocations")

    agent_list = list(agents)
    n = len(agent_list)

    for r in range(1, n + 1):
        for agent_subset in combinations(agent_list, r):
            agent_bundle_options = []
            skip_subset = False

            for agent in agent_subset:
                prefs = tuple(sorted(preferences[agent]))
                bundles = generate_agent_bundles(prefs, demands[agent])
                if not bundles:
                    skip_subset = True
                    break
                agent_bundle_options.append(bundles)

            if skip_subset:
                continue

            for joint in product(*agent_bundle_options):
                item_total = Counter()
                used_items = set()
                allocation = []

                valid = True
                for agent, bundle in zip(agent_subset, joint):
                    if not bundle:
                        continue

                    item = bundle[0]  # Since bundle is like ('a', 'a')
                    if item in used_items:
                        valid = False
                        break
                    if len(set(bundle)) > 1:
                        valid = False
                        break

                    item_total[item] += len(bundle)
                    if item_total[item] > capacities[item]:
                        valid = False
                        break

                    used_items.add(item)
                    allocation.extend((agent, item) for _ in bundle)

                if valid:
                    yield (frozenset(agent_subset), allocation)



def leximin_primal(alloc: AllocationBuilder) -> None:
    """
    Algorithm 2: Leximin Primal — computes a fair allocation of classrooms from public schools
     to charter schools, aiming to maximize the satisfaction of the least advantaged agent.

    Example 1: Two agents, disjoint desires
    >>> from fairpyx.instances import Instance
    >>> from fairpyx.allocations import AllocationBuilder
    >>> from fairpyx.algorithms.leximin_primal import leximin_primal
    >>> instance = Instance(
    ...     valuations={1: {"a": 1}, 2: {"b": 1}},
    ...     agent_capacities={1: 1, 2: 1},
    ...     item_capacities={"a": 1, "b": 1}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> leximin_primal(alloc)
    >>> expected = [{1: {"a": 1}, 2: {"b": 1}}]
    >>> actual = [a for a, _ in alloc.distribution]
    >>> actual == expected
    True
    >>> probs = [p for _, p in alloc.distribution]
    >>> len(probs) == 1 and abs(probs[0] - 1.0) < 1e-6
    True

    Example 2: One item, two agents — one agent gets it
    >>> from fairpyx.instances import Instance
    >>> from fairpyx.allocations import AllocationBuilder
    >>> from fairpyx.algorithms.leximin_primal import leximin_primal

    >>> instance = Instance(
    ...     valuations={1: {"a": 1}, 2: {"a": 1}},  # Both agents want same item
    ...     agent_capacities={1: 1, 2: 1},
    ...     item_capacities={"a": 1}  # Only one unit available
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> leximin_primal(alloc)
    >>> expected = [{1: {"a": 1}, 2: {}}, {1: {}, 2: {"a": 1}}]
    >>> actual = [a for a, _ in alloc.distribution]

    >>> all(a in expected for a in actual) and len(actual) == 2
    True
    >>> total_prob = sum(p for _, p in alloc.distribution)
    >>> abs(total_prob - 1.0) < 1e-6
    True

    Example 3: One agent gets one item
    >>> instance = Instance(
    ...     valuations={1: {"a": 1}},
    ...     agent_capacities={1: 1},
    ...     item_capacities={"a": 1}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> leximin_primal(alloc)
    >>> expected = [{1: {"a": 1}}]
    >>> actual = [a for a, _ in alloc.distribution]
    >>> actual == expected
    True
    >>> probs = [p for _, p in alloc.distribution]
    >>> len(probs) == 1 and abs(probs[0] - 1.0) < 1e-6
    True

    Example 4: Agents and items with capacities > 1
    >>> instance = Instance(
    ...     valuations={1: {"a": 1, "b": 1}, 2: {"a": 1, "c": 1}},
    ...     agent_capacities={1: 2, 2: 3},
    ...     item_capacities={"a": 3, "b": 1, "c": 2}
    ... )
    >>> alloc = AllocationBuilder(instance)
    >>> leximin_primal(alloc)
    >>> total_prob = sum(p for _, p in alloc.distribution)
    >>> abs(total_prob - 1.0) < 1e-6
    True
    """
    # === Input: {(di, Fi)} for i ∈ N and {cj} for j ∈ M ===

    # Step 0: Preprocessing – extract relevant structures from the instance
    agents = list(alloc.remaining_agent_capacities.keys())
    items = list(alloc.remaining_item_capacities.keys())

    if not agents:
        logger.info("No agents provided. Nothing to allocate.")
        alloc.distribution = []
        return

    if not items:
        logger.info("No items with capacity. Cannot allocate anything.")
        alloc.distribution = []
        return

    demands = alloc.remaining_agent_capacities
    capacities = alloc.remaining_item_capacities

    # Construct Fi = {items with positive value to agent i}
    valuations = {
        agent: {
            item: alloc.instance.agent_item_value(agent, item)
            for item in items
            if alloc.instance.agent_item_value(agent, item) > 0
        }
        for agent in agents
    }
    preferences = {i: set(v.keys()) for i, v in valuations.items()}

    # Remove agents with no preferences
    agents = [i for i in agents if preferences[i]]
    if not agents:
        logger.info("No agents with non-empty preferences. Nothing to allocate.")
        alloc.distribution = []
        return

    # Edge case: all agents have empty preference sets
    if all(len(prefs) == 0 for prefs in preferences.values()):
        logger.info("All agents have empty preference sets. No feasible allocations.")
        alloc.distribution = []
        return

    logger.info("=== Solving FeasibilityILP ===")

    # === Line 1-2: Generate feasible (S, AS) pairs using ILP with pruning ===
    feasible_sets = list(generate_feasible_sets(agents, items, demands, capacities, preferences))
    logger.info(f"Total feasible allocations: {len(feasible_sets)}")

    # === Line 3: R ← N ===
    R = set(agents)

    # === Line 4: p*_i ← 0 for all i ∈ N ===
    p_star = {i: 0.0 for i in agents}

    # Accumulate allocation probabilities over iterations
    xS_total = {tuple(alloc): 0.0 for _, alloc in feasible_sets}

    # === Line 5–8: Iterate while R ≠ ∅ ===
    iteration = 0
    while R:
        iteration += 1
        logger.info(f"\n=== Iteration {iteration} ===")

        # === Line 6: Solve PrimalLP to get M, {pi}, {xS} ===
        M, pi, xS = primal_lp(feasible_sets, R, agents, p_star)

        # Accumulate allocation probabilities into xS_total
        for S in xS:
            val = value(xS[S])

            # We use += instead of = because the same allocation S may receive positive weight across multiple iterations.
            # Using = would overwrite previous probability mass, breaking the overall distribution.
            xS_total[S] += val

        # === Line 7–8: Fix agents i ∈ R such that pi = M, update p*_i ← M and R ← R \ {i} ===
        for i in list(R):
            if abs(pi[i] - value(M)) < 1e-6:
                # Set p*_i ← M per Algorithm
                p_star[i] = value(M)
                R.remove(i)

    # === Line 9–10: R is now empty — decide output strategy ===
    if abs(value(M) - 1.0) < 1e-6:
        from collections import Counter

        alloc.distribution = []
        threshold = max(len(alloc_tuple) for alloc_tuple, weight in xS_total.items() if weight > 1e-6)

        for alloc_tuple, weight in xS_total.items():
            if weight < 1e-6:
                continue

            if len(alloc_tuple) < threshold:
                continue

            temp_alloc = AllocationBuilder(alloc.instance)
            temp_alloc.set_allow_multiple_copies(True)

            for (i, j) in alloc_tuple:
                try:
                    temp_alloc.give(i, j)
                except ValueError:
                    logger.warning(f"Could not assign {j} to {i} (capacity full)")

            for agent in agents:
                temp_alloc.bundles.setdefault(agent, {})

            normalized_bundle = {}
            for agent, bundle in temp_alloc.bundles.items():
                counts = Counter(bundle)
                normalized_bundle[agent] = dict(counts)

            alloc.distribution.append((normalized_bundle, weight))

        total_prob = sum(prob for _, prob in alloc.distribution)
        if total_prob > 0:
            alloc.distribution = [
                (bundle, prob / total_prob)
                for bundle, prob in alloc.distribution
            ]

        logger.info("\n=== Final Allocation Results Summary ===")
        for allocation, prob in alloc.distribution:
            logger.info(f"Allocation: {allocation} with Probability: {prob:.4f}")

    else:
        alloc.distribution = []
        for S in xS_total:
            prob_val = xS_total[S]
            if prob_val < 1e-6:
                continue

            temp_alloc = AllocationBuilder(alloc.instance)
            temp_alloc.set_allow_multiple_copies(True)
            for (i, j) in S:
                try:
                    temp_alloc.give(i, j)
                except ValueError:
                    logger.warning(f"Could not assign {j} to {i} (capacity full)")

            for agent in agents:
                temp_alloc.bundles.setdefault(agent, {})

            from collections import Counter

            normalized_bundle = {}
            for agent, bundle in temp_alloc.bundles.items():
                counts = Counter(bundle)
                normalized_bundle[agent] = dict(counts)

            alloc.distribution.append((normalized_bundle, prob_val))

        total_prob = sum(prob for _, prob in alloc.distribution)
        if total_prob > 0:
            alloc.distribution = [
                (bundle, prob / total_prob)
                for bundle, prob in alloc.distribution
            ]

        logger.info("\n=== Final Allocation Results Summary ===")
        for allocation, prob in alloc.distribution:
            logger.info(f"Allocation: {allocation} with Probability: {prob:.4f}")


if __name__ == "__main__":
    #import doctest
    import logging

    #doctest.testmod(verbose=True)
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

    from fairpyx.instances import Instance
    from fairpyx.allocations import AllocationBuilder

    # Example: 3 agents, 2 facilities
    instance = Instance(
        valuations={1: {"a": 1, "b": 1}, 2: {"a": 1}, 3: {"a": 1, "b": 1}},  # F1, F2, F3
        agent_capacities={1: 3, 2: 1, 3: 1},
        item_capacities={"a": 3, "b": 2}
    )
    alloc = AllocationBuilder(instance)

    # Run LeximinPrimal
    leximin_primal(alloc)

    # Output final distribution summary
    print("\n=== Final Allocation Summary ===")
    for i, (allocation, prob) in enumerate(alloc.distribution, 1):
        print(f"Allocation {i}:")
        for agent, bundle in allocation.items():
            print(f"  Agent {agent} receives: {bundle}")
        print(f"  Probability: {prob:.4f}")