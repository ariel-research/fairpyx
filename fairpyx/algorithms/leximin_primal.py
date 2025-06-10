"""
An implementation of the algorithm in:
"Leximin Allocations in the Real World", by D. Kurokawa, A. D. Procaccia, and N. Shah (2018), https://doi.org/10.1145/3274641

Programmer: Lior Trachtman
Date: 2025-05-05
"""

import logging
from fairpyx.allocations import AllocationBuilder
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpContinuous, value, PULP_CBC_CMD
from itertools import combinations

from pulp import LpProblem, LpVariable, LpBinary, lpSum, LpMaximize

# Setup basic logging configuration (you can customize format and level)
logger = logging.getLogger(__name__)

def feasibility_ilp(S, items, demands, capacities, preferences):
    """
    Solve the FeasibilityILP subproblem for a given subset of agents S.
    Returns:
        A list of (agent, facility) pairs indicating deterministic assignment,
        or None if no feasible assignment exists.

    Based on the algorithm:
        ∑_{f ∈ Fi} y_{i,f} ≥ 1       for all i ∈ S
        ∑_{i ∈ S : f ∈ Fi} d_i·y_{i,f} ≤ c_f  for all f ∈ M
        y_{i,f} ∈ {0, 1}             for all i ∈ S, f ∈ Fi

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

    >>> # Infeasible case: both want 'a', but capacity is 1
    >>> S = {1, 2}
    >>> items = {'a'}
    >>> demands = {1: 1, 2: 1}
    >>> capacities = {'a': 1}
    >>> preferences = {1: {'a'}, 2: {'a'}}
    >>> feasibility_ilp(S, items, demands, capacities, preferences) is None
    True
    """

    logger.info(f"Starting FeasibilityILP for agents subset: {S}")

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
        logger.debug(f"Added constraint: agent {i} assigned at least one preferred facility")

    # === Constraint 2: ∑_{i ∈ S : f ∈ Fi} d_i * y_{i,f} ≤ c_f for all f ∈ M ===
    # Total assigned demand for each facility must not exceed its capacity
    for f in items:
        prob += lpSum(
            demands[i] * y[i, f]
            for i in S
            if f in preferences[i]
        ) <= capacities[f]
        logger.debug(f"Added capacity constraint for facility '{f}': sum demand ≤ {capacities[f]}")

    # === Solve the ILP ===
    logger.info("Solving FeasibilityILP...")
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
    logger.info("Starting PrimalLP problem setup")

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
    logger.debug("Added constraint: sum of xS variables equals 1")

    # === Constraints for each agent i ∈ N ===
    for i in agents:
        # pi = sum of xS for all allocations where agent i appears
        pi_expr = lpSum(xS[S] for S in xS if any(s[0] == i for s in S))
        if i in R:
            prob += pi_expr >= M
            logger.debug(f"Added constraint for agent {i}: pi >= M")
        else:
            prob += pi_expr == p_star[i]
            logger.debug(f"Added constraint for agent {i}: pi == p_star[{i}] = {p_star[i]}")

    # === Final objective: maximize M ===
    prob += M
    logger.info("Set objective to maximize M")

    # === Solve LP ===
    logger.info("Solving PrimalLP...")
    prob.solve(PULP_CBC_CMD(msg=0))


    if prob.status != 1:
        logger.error(f"Primal LP is infeasible, status code: {prob.status}")
        raise Exception("Primal LP is infeasible")

    # === Compute actual values of pi for i ∈ R ===
    pi = {}
    for i in R:
        pi_val = sum(value(xS[S]) for S in xS if any(s[0] == i for s in S))
        pi[i] = pi_val
        logger.debug(f"Computed pi[{i}] = {pi_val:.6f}")

    M_val = value(M)
    logger.info(f"PrimalLP solved successfully: M = {M_val:.6f}")

    return M_val, pi, xS


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

    # Edge case: all agents have empty preference sets
    if all(len(prefs) == 0 for prefs in preferences.values()):
        logger.info("All agents have empty preference sets. No feasible allocations.")
        alloc.distribution = []
        return

    # === Line 1-2: Solve FeasibilityILP for each subset S ⊆ N, For each S ∈ S, let AS be the returned assignment implicit via feasible_sets = list of (S, AS)===
    feasible_sets = []
    failed_subsets = set()
    logger.info("\n=== Generating feasible allocations (FeasibilityILP) with pruning ===")

    for r in range(1, len(agents) + 1):
        for subset in combinations(agents, r):
            subset_frozen = frozenset(subset)

            # Prune if any known failed subset is a subset of this one
            if any(failed.issubset(subset_frozen) for failed in failed_subsets):
                continue

            # Quick precheck: total demand vs total capacity
            total_demand = sum(demands[i] for i in subset)
            total_capacity = sum(capacities[f] for f in items)
            if total_demand > total_capacity:
                continue

            alloc_set = feasibility_ilp(subset, items, demands, capacities, preferences)
            if alloc_set:
                logger.debug(f"Feasible allocation found for {subset}: {alloc_set}")
                feasible_sets.append((subset, alloc_set))
            else:
                failed_subsets.add(subset_frozen)

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
        logger.info(f"\n--- Iteration {iteration} ---")

        # === Line 6: Solve PrimalLP to get M, {pi}, {xS} ===
        M, pi, xS = primal_lp(feasible_sets, R, agents, p_star)
        logger.info(f"M = {value(M):.6f}")

        # Accumulate allocation probabilities into xS_total
        for S in xS:
            val = value(xS[S])
            xS_total[S] += val
            if val > 1e-6:
                logger.debug(f"xS[{S}] += {val:.4f}")

        # === Line 7–8: Fix agents i ∈ R such that pi = M, update p*_i ← M and R ← R \ {i} ===
        for i in list(R):
            pi_val = sum(value(xS[S]) for S in xS if any(s[0] == i for s in S))
            if abs(pi_val - value(M)) < 1e-6:
                p_star[i] = pi_val
                logger.info(f"Fixing p*[{i}] = {pi_val:.4f}")
                R.remove(i)

    # === Line 9–10: R is now empty — return randomized allocation using xS_total ===
    logger.info("\n=== Final Allocation Probabilities ===")
    for S, prob_val in xS_total.items():
        if prob_val > 0:
            logger.info(f"Bundle {S} with prob {prob_val:.4f}")

    logger.info("\n=== Exploring All Possible Allocations from the Distribution ===")
    population = list(xS_total.keys())
    alloc.distribution = []  # reset previous allocations

    # Build and normalize output distribution
    for alloc_idx, S in enumerate(population):
        if xS_total[S] < 1e-6:
            continue  # skip near-zero allocations

        logger.info(f"\n--- Allocation {alloc_idx + 1} ---")
        logger.info(f"Probability: {xS_total[S]:.4f}")

        # Simulate allocation S
        temp_alloc = AllocationBuilder(alloc.instance)
        temp_alloc.set_allow_multiple_copies(True)

        for (i, j) in S:
            logger.debug(f"Assigning item {j} to agent {i}")
            try:
                temp_alloc.give(i, j)
            except ValueError:
                logger.warning(f"Could not assign {j} to {i} (capacity full)")

        logger.info("--- Resulting Allocation ---")
        for agent, bundle in temp_alloc.bundles.items():
            logger.info(f"{agent}: {bundle}")

        # Ensure all agents appear, even if empty
        for agent in agents:
            temp_alloc.bundles.setdefault(agent, {})

        # Format: {agent: {item: 1}, ...}
        normalized_bundle = {
            agent: {str(item): 1 for item in bundle}
            for agent, bundle in temp_alloc.bundles.items()
        }

        # Save result
        alloc.distribution.append((normalized_bundle, xS_total[S]))

    # Normalize total probability to sum to 1.0
    total_prob = sum(prob for _, prob in alloc.distribution)
    if total_prob > 0:
        alloc.distribution = [
            (bundle, prob / total_prob)
            for bundle, prob in alloc.distribution
        ]


if __name__ == "__main__":
    import doctest
    import logging

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    doctest.testmod(verbose=True)
