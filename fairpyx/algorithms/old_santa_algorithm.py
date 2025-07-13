"""
An implementation of the algorithms in:
"Santa Claus Meets Hypergraph Matchings",
by ARASH ASADPOUR - New York University, URIEL FEIGE - The Weizmann Institute, AMIN SABERI - Stanford University,
https://dl.acm.org/doi/abs/10.1145/2229163.2229168
Programmers: May Rozen
Date: 2025-04-23
"""
 # The paper presents a single algorithm, which is built in a modular way.
 # Similarly, in the fairpyx library for fair division algorithms, the algorithms are also structured modularly.
 # Therefore, I structured the algorithm headers here in the same manner.

# ------------- This is the algo without cache -------------

from typing import Dict, List, Set, Tuple
from hypernetx import Hypergraph as HNXHypergraph
from fairpyx import Instance, AllocationBuilder
import logging
from typing import Optional
import cvxpy as cp
import itertools
from itertools import combinations
import time

# Logger definition
logger = logging.getLogger(__name__)


def parse_allocation_strings(allocation: Dict[str, str]) -> Dict[str, List[Set[str]]]:
    """
    Receives an allocation in a string format, such as: "1.0*{c1, c3}" או "1.0*{'c1', 'c3'}"
    and returns the allocation in a structured format: {'Alice': [{'c1', 'c3'}], ...}

    If there is no bundle (e.g., "0.0*{}"), or if there is a formatting error, an empty list is returned
    """
    parsed: Dict[str, List[Set[str]]] = {}
    for agent, bundle_str in allocation.items():
        parsed[agent] = []  # Default: no bundles

        # Check that there is a substring starting with *{ and ending with }
        if "*{" not in bundle_str or "}" not in bundle_str:
            continue

        # Extract the part that comes after the asterisk (*) until the end of the string
        bundle_part = bundle_str.split("*", 1)[1].strip()

        # If there are no curly braces {} at all – continue with an empty list
        if not (bundle_part.startswith("{") and bundle_part.endswith("}")):
            continue

        # Remove the curly braces {}
        inner = bundle_part[1:-1].strip()
        if inner == "":
            # If the bundle is empty, leave: parsed[agent] = []
            continue

        # Split the string by commas
        items = []
        for token in inner.split(","):
            tok = token.strip()
            # Safely remove any surrounding quotation marks from the string
            if len(tok) >= 2 and ((tok.startswith("'") and tok.endswith("'")) or (tok.startswith('"') and tok.endswith('"'))):
                tok = tok[1:-1]
            if tok != "":
                items.append(tok)

        if items:
            parsed[agent] = [set(items)]

    return parsed


def old_santa_claus_main(allocation_builder: AllocationBuilder) -> Dict[str, Set[str]]:
    """
    The main function implements the Santa Claus algorithm:
    1. Performs binary search over the threshold t.
    2. Solves a linear program (LP) and obtains a fractional allocation.
    3. Classifies items as fat or thin.
    4. Constructs a hypergraph.
    5. Performs local search to find a perfect matching.
    6. Produces the final allocation under capacity constraints.
    Returns a dictionary of the form {agent: set(items)}.

    >>> # Test 1: Simple case with 2 players and 3 items
    >>> instance = Instance(
    ...     valuations={"Alice": {"c1": 5, "c2": 0, "c3": 6}, "Bob": {"c1": 0, "c2": 8, "c3": 0}},
    ...     agent_capacities={"Alice": 2, "Bob": 1},
    ...     item_capacities={"c1": 5, "c2": 8, "c3": 6},
    ... )
    >>> allocation_builder = AllocationBuilder(instance=instance)
    >>> result = old_santa_claus_main(allocation_builder)
    >>> result == {'Alice': {'c1', 'c3'}, 'Bob': {'c2'}}
    True

    >>> # Test 2: More complex case with 4 players and 4 items
    >>> instance = Instance(
    ...     valuations={"A": {"c1": 10, "c2": 0, "c3": 0, "c4": 0}, "B": {"c1": 0, "c2": 8, "c3": 0, "c4": 0}, "C": {"c1": 0, "c2": 0, "c3": 10, "c4": 0}, "D": {"c1": 0, "c2": 0, "c3": 0, "c4": 6}},
    ...     agent_capacities={"A": 1, "B": 1, "C": 1, "D": 1},
    ...     item_capacities={"c1": 1, "c2": 1, "c3": 1, "c4": 1},
    ... )
    >>> allocation_builder = AllocationBuilder(instance=instance)
    >>> result = old_santa_claus_main(allocation_builder)
    >>> result == {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
    True

    >>> # Test 3: A receives two gifts, and B receives one
    >>> instance = Instance(
    ...     valuations={"A": {"c1": 5, "c2": 5, "c3": 0}, "B": {"c1": 0, "c2": 0, "c3": 6}},
    ...     agent_capacities={"A": 2, "B": 1},
    ...     item_capacities={"c1": 1, "c2": 1, "c3": 1},
    ... )
    >>> allocation_builder = AllocationBuilder(instance=instance)
    >>> result = old_santa_claus_main(allocation_builder)
    >>> result == {'A': {'c1', 'c2'}, 'B': {'c3'}}
    True

    >>> # Test 4: There are more agents than items – an error is raised
    >>> try:
    ...     instance = Instance(
    ...         valuations={"A": {"c1": 10}, "B": {"c1": 0}, "C": {"c1": 0}},
    ...         agent_capacities={"A": 1, "B": 1, "C": 1},
    ...         item_capacities={"c1": 1}
    ...     )
    ...     allocation_builder = AllocationBuilder(instance=instance)
    ...     old_santa_claus_main(allocation_builder)
    ... except ValueError as e:
    ...     print("Caught expected exception:", e)
    Caught expected exception: Too few items for the number of agents: 1 items for 3 agents.

    """
    start = time.perf_counter()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    # Extract information from the AllocationBuilder: agent and item names
    instance = allocation_builder.instance
    agent_names = list(instance.agents)
    item_names = list(instance.items)
    agent_capacities = {a: instance.agent_capacity(a) for a in agent_names}
    if len(agent_names) > len(item_names): # Raise an error if there are more agents than items
        raise ValueError(f"Too few items for the number of agents: {len(item_names)} items for {len(agent_names)} agents.")

    logger.info("Starting santa_claus_main")
    logger.debug("Instance agents: %s", agent_names)
    logger.debug("Instance items: %s", item_names)

    # Build the valuation matrix: for each agent, what is their value for each item
    valuations = {
        agent: {
            item: instance.agent_item_value(agent, item)
            for item in item_names
        }
        for agent in agent_names
    }

    # Compute the binary search range based on the maximum value of any item
    high = min(sum(v.values()) for v in valuations.values()) # high = min [over all agents i] of sum[all_item_values for i]

    low = 0
    logger.debug("Initial valuations: %s", valuations)
    logger.debug("Initial binary search range: low=%f, high=%f", low, high)

    best_matching = {}

    # == Binary search on t ==
    # Binary search is limited to 10 steps with precision 1e-4
    for step in range(1, 11):
        mid = (low + high) / 2

        # Clear separation and logging for each step
        logger.info("\n\n==== Binary search step %d: t=%.4f (low=%.4f, high=%.4f) ====",
                    step, mid, low, high)

        feasible, matching = is_threshold_feasible(valuations, mid, agent_names)
        if feasible:
            best_matching = matching
            low = mid
            logger.info("Threshold %.4f feasible: matching %s", mid, matching)
        else:
            high = mid
            logger.info("Threshold %.4f infeasible", mid)

        # Early stopping once the desired precision is reached
        if high - low <= 1e-4:
            logger.info("Desired precision (1e-4) reached after %d steps", step)
            break

    logger.info("Binary search completed after %d steps: final threshold t≈%.4f", step, low)

    used_items = set()
    for agent, items in best_matching.items():
        cap = agent_capacities.get(agent, 1)
        # only up to 'cap' items
        for item in list(items)[:cap]:
            if item in used_items:
                continue  # skip if already allocated
            allocation_builder.give(agent, item)
            used_items.add(item)

    logger.info("Final matching found at threshold %.4f: %s", low, best_matching)
    end = time.perf_counter()
    logger.info("santa_claus_main took %.4f seconds", end - start)
    return best_matching

def is_threshold_feasible(valuations: Dict[str, Dict[str, float]], threshold: float, agent_names: List[str]) -> Tuple[bool, Dict[str, str]]:
    """
    Checks whether there exists an assignment in which every player receives a bundle
    whose total value is at least the given threshold.

    Explanation:
    This is step 1 of the algorithm – Threshold Selection.
    We choose a threshold value t and try to determine whether there exists an allocation
    in which each player receives a bundle worth at least t.
    Later, the algorithm performs binary search on t to find the maximum feasible threshold.
    This function helps determine whether, for a given t, there exists a valid allocation that satisfies all players.

    Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "Alice": {"c1": 7, "c2": 0, "c3": 4},
    ...     "Bob":   {"c1": 0,  "c2": 8, "c3": 0}
    ... }
    >>> is_threshold_feasible(valuations, 15,{"Alice","Bob"})[0]
    False
    >>> is_threshold_feasible(valuations, 10,{"Alice","Bob"})[0]
    False
    >>> is_threshold_feasible(valuations, 8,{"Alice","Bob"})[0]
    True

    Example 2: 2 Players, 2 Items (conflict)
    >>> valuations = {
    ...     "Alice": {"c1": 10, "c2": 0},
    ...     "Bob":   {"c1": 0, "c2": 9}
    ... }
    >>> is_threshold_feasible(valuations, 10,{"Alice","Bob"})[0]
    False
    >>> is_threshold_feasible(valuations, 9,{"Alice","Bob"})[0]
    True
    """

    # Solve the linear program to obtain an initial allocation
    raw_allocation = solve_configuration_lp(valuations, threshold)
    allocation = parse_allocation_strings(raw_allocation)  # Convert format
    fat_items, thin_items = classify_items(valuations, threshold)  # Classify items as fat or thin based on the threshold
    H = build_hypergraph(valuations, allocation, fat_items, thin_items, threshold)  # Build a hypergraph from the allocation

    matching = local_search_perfect_matching(H, valuations, agent_names,
                                             threshold=threshold)  # Perform local search to find a perfect matching
    if len(matching) == len(agent_names):  # If the matching size equals the number of agents/players
        for player, items in valuations.items():
            total_value = sum(value for value in items.values())

            if total_value < threshold:
                return False, {}

        logger.info("Threshold feasibility check passed, all players can receive at least %f value", threshold)
        return True, matching

    return False, {}

def solve_configuration_lp(valuations: Dict[str, Dict[str, float]], threshold: float) -> Dict[str, str]:
    """
    Explanation:
    This is step 2 of the algorithm – Configuration LP Relaxation.
    We define binary variables x_{i,S} indicating whether player i receives a bundle S ⊆ R
    whose total value is at least t/4.
    This function solves the relaxed linear program (allowing fractional x_{i,S}) and returns,
    for each player, a set of such bundles with positive fractional weight.
    Although the resulting allocation is not integral, it will be used later to construct
    the hypergraph for the matching stage.

    Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "Alice": {"c1": 7, "c2": 0, "c3": 7},
    ...     "Bob":   {"c1": 0,  "c2": 8, "c3": 0}
    ... }
    >>> solve_configuration_lp(valuations, 8)
    {'Alice': '1.0*{c1, c3}', 'Bob': '1.0*{c2}'}

    """
    logger.info("Solving configuration LP with cvxpy at threshold %.4f", threshold)

    agents = list(valuations.keys())
    items = sorted({item for v in valuations.values() for item in v.keys()})

    # All possible configurations (bundles) for each agent
    bundles = {
        i: [frozenset(s) for r in range(1, len(items) + 1)
            for s in itertools.combinations(items, r)
            if sum(valuations[i].get(x, 0) for x in s) >= threshold]
        for i in agents
    }

    # LP variables: x_{i,S}
    x = {
        (i, S): cp.Variable(nonneg=True)
        for i in agents
        for S in bundles[i]
    }

    constraints = []

    # Constraint 1: Each agent can receive at most one bundle
    for i in agents:
        constraints.append(cp.sum([x[i, S] for S in bundles[i]]) <= 1)

    # Constraint 2: Each item can be assigned at most once
    for j in items:
        terms = []
        for i in agents:
            for S in bundles[i]:
                if j in S:
                    terms.append(x[i, S])
        if terms:
            constraints.append(cp.sum(terms) <= 1)

    # Objective: Arbitrary maximization (we only care about feasibility)
    objective = cp.Maximize(cp.sum(list(x.values())))

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)

    # Construct output: for each agent, choose the bundle with the highest fractional value
    allocation = {}
    for i in agents:
        max_val = 0
        max_bundle = frozenset()
        for S in bundles[i]:
            val = x[i, S].value
            if val is not None and val > max_val:
                max_val = val
                max_bundle = S
        if max_val > 0:
            allocation[i] = f"{round(max_val, 4)}*{{{', '.join(sorted(max_bundle))}}}"
        else:
            allocation[i] = "0.0*{}"

    logger.debug("Configuration LP solution: %s", allocation)
    return allocation


def classify_items(valuations: Dict[str, Dict[str, float]], threshold: float) -> Tuple[Set[str], Set[str]]:
    """
    Classifies items as fat if their value to any individual player is ≥ t/4,
    or thin if the value is less than t/4.

    Explanation:
    This is step 3 of the algorithm – item classification after normalization.
    We normalize the threshold so that t = 1.
    Any item worth at least 1/4 to some player is considered fat; all others are considered thin.
    The goal is to reduce complexity and limit the bundle sizes in the hypergraph.
    Later, we will only construct minimal hyperedges that satisfy this condition.

    Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "Alice": {"c1": 0.5, "c2": 0, "c3": 0},
    ...     "Bob":   {"c1": 0, "c2": 0.1, "c3": 0.2}
    ... }
    >>> fat, thin = classify_items(valuations, 1)
    >>> fat == {'c1'} and thin == {'c2', 'c3'}
    True
    """
    fat_items, thin_items = set(), set() # Empty sets: one for fat items, one for thin items
    for item in next(iter(valuations.values())).keys(): # Extract item names from the valuations dictionary
        max_val = max(agent_val[item] for agent_val in valuations.values()) # Compute the maximum value of this item across all players

        # Classification decision: fat or thin?
        # If any player values the item at least threshold / 4,
        # then the item is considered "fat" – based on the paper's condition
        # ensuring that a single item can contribute at least one-quarter of the threshold t.
        if max_val >= threshold / 4:
            fat_items.add(item)
        else:
            thin_items.add(item)

    logger.info("Classifying items with threshold %.4f", threshold)
    logger.debug("Fat items: %s", fat_items)
    logger.debug("Thin items: %s", thin_items)
    return fat_items, thin_items

def build_hypergraph(valuations: Dict[str, Dict[str, float]],
                         allocation: Dict[str, List[Set[str]]],
                         fat_items: Set[str],
                         thin_items: Set[str],
                         threshold: float) -> HNXHypergraph:
    """
    Builds a bipartite hypergraph where hyperedges represent fat or thin bundles
    with total value at least the given threshold.

    Explanation:
    This is step 4 of the algorithm – hypergraph construction.
    We build a hypergraph where:
    - One side contains the players.
    - The other side contains the items.
    - Each fat or thin bundle from the fractional allocation becomes a hyperedge.
    Specifically:
    - For each fat item assigned to a player, we create a hyperedge {i, j}.
    - For each thin bundle, we ensure that the removal of any item would
      cause the bundle's value to drop below 1 — meaning all items are essential.

    The goal of the hypergraph is to enable a search for a perfect matching in the next step.

    Example: 4 Players, 4 Items
    >>> valuations = {
    ...     "A": {"c1": 10, "c2": 0, "c3": 0, "c4": 0},
    ...     "B": {"c1": 0,  "c2": 8, "c3": 0, "c4": 0},
    ...     "C": {"c1": 0,  "c2": 0, "c3": 6, "c4": 0},
    ...     "D": {"c1": 0,  "c2": 0, "c3": 0, "c4": 4}
    ... }
    >>> allocation = {
    ...     "A": [{"c1"}],
    ...     "B": [{"c2"}],
    ...     "C": [{"c3"}],
    ...     "D": [{"c4"}]
    ... }
    >>> fat_items, thin_items = classify_items(valuations, 4)
    >>> hypergraph = build_hypergraph(valuations, allocation, fat_items, thin_items, 4)
    >>> len(hypergraph.nodes)  # Number of nodes
    8
    >>> len(hypergraph.edges)  # Number of edges
    4
    """
    logger.info("Building hypergraph based on allocation")

    edges: dict[str, set[str]] = {}
    edge_id = 0
    seen: set[frozenset[str]] = set()

    # 1. Add edges from Configuration LP ("lp*" edges)
    for player, bundles in allocation.items():
        for bundle in bundles:
            nodes = frozenset({player, *bundle})
            if nodes in seen:
                continue
            seen.add(nodes)
            edges[f"lp{edge_id}"] = set(nodes)
            edge_id += 1

    # 2. Add fat edges: only if the player values the item ≥ threshold
    for item in fat_items:
        for player in valuations:
            if valuations[player].get(item, 0) >= threshold:
                nodes = frozenset({player, item})
                if nodes in seen:
                    continue
                seen.add(nodes)
                edges[f"f{edge_id}"] = set(nodes)
                edge_id += 1

    # 3. Add thin edges: for each minimal subset of thin items whose total value for the player is ≥ threshold
    for player in valuations:
        for r in range(1, len(thin_items) + 1):
            for bundle in combinations(thin_items, r):
                total = sum(valuations[player].get(i, 0) for i in bundle)
                if total < threshold:
                    continue

                is_minimal = True
                for x in bundle:
                    if total - valuations[player].get(x, 0) >= threshold:
                        is_minimal = False
                        break

                if not is_minimal:
                    continue

                nodes = frozenset({player, *bundle})
                if nodes in seen:
                    continue
                seen.add(nodes)
                edges[f"t{edge_id}"] = set(nodes)
                edge_id += 1

    H = HNXHypergraph(edges)

    edge_strs = []
    for edge in H.edges:
        # 1) build the comma‐separated list of quoted node names:
        nodes_list = ", ".join(f'"{node}"' for node in H.edges[edge])
        # 2) wrap it in braces and prepend the edge name:
        edge_strs.append(f'"{edge}": {{{nodes_list}}}')

    # now join all of those:
    edges_repr = ", ".join(edge_strs)

    logger.info(
        "Hypergraph construction completed with %d nodes and %d edges: {%s}",
        len(H.nodes),
        len(H.edges),
        edges_repr
    )

    return H

# Helper function – returns an edge that can be added to the alternating tree
def extend_alternating_tree(H: HNXHypergraph,
                            visited_players: Set[str],
                            visited_edges: Set[str],
                            players: List[str],
                            valuations: Dict[str, Dict[str, float]],
                            threshold: float) -> Optional[str]:
    """
    Attempts to extend the alternating tree according to Lemma 3.2.
    Returns the name of an edge that can be added to the tree, or None if none is found.
    """

    covered_nodes = set() # All nodes currently in the tree (both players and items)
    for edge_name in visited_edges: # All edges already in the tree
        covered_nodes |= set(H.edges[edge_name])
    covered_items = covered_nodes - set(players) # Only items (we’re looking for new ones)


    for edge_name in H.edges: # Iterate over all edges not yet used
        if edge_name in visited_edges:
            continue # Skip edges already visited
        edge_nodes = set(H.edges[edge_name])
        edge_players = edge_nodes & set(players) # Players in this edge
        edge_items = edge_nodes - set(players) # Items in this edge

        if not edge_players & visited_players: # Skip if the edge is not connected to the tree
            continue

        # Skip if the edge does not introduce any new item
        if not edge_items.isdisjoint(covered_items):
            continue

        # Check if the edge satisfies any player
        for player in edge_players:
            value = sum(valuations[player].get(item, 0) for item in edge_items)
            if value >= threshold: # If the total value meets the threshold
                return edge_name

    return None

def local_search_perfect_matching(H: HNXHypergraph, valuations: Dict[str, Dict[str, float]], players: List[str], threshold: float) -> Dict[str, Set[str]]:
    """
    Performs local search to find a perfect matching in the hypergraph —
    each player is matched to a separate bundle whose value is at least the threshold.

    Explanation:
    This is step 5 of the algorithm – local search for a perfect matching in the hypergraph.
    The algorithm incrementally builds a perfect matching using alternating trees and augmenting edges.
    The idea is to start from an empty matching and expand it step by step:
    - Select an unmatched player.
    - Construct an alternating tree rooted at that player.
    - Search for an augmenting edge (non-blocking) to extend the matching.

    The algorithm guarantees that each player receives a disjoint bundle
    with total value at least t.

     Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "A": {"c1": 5, "c2": 0, "c3": 4, "c4": 0},
    ...     "B": {"c1": 5, "c2": 6, "c3": 0, "c4": 0},
    ...     "C": {"c1": 0, "c2": 6, "c3": 4, "c4": 0},
    ...     "D": {"c1": 0, "c2": 0, "c3": 4, "c4": 6}
    ... }
    >>> threshold = 4
    >>> fat_items, thin_items = classify_items(valuations, threshold)
    >>> print(fat_items == {'c1', 'c2', 'c3', 'c4'})
    True


    >>> from hypernetx import Hypergraph as HNXHypergraph
    >>> edge_dict = {
    ...     "A_c1": {"A", "c1"},
    ...     "A_c3": {"A", "c3"},
    ...     "A_c1c3": {"A", "c1", "c3"},
    ...     "B_c1": {"B", "c1"},
    ...     "B_c2": {"B", "c2"},
    ...     "B_c1c2": {"B", "c1", "c2"},
    ...     "C_c2": {"C", "c2"},
    ...     "C_c3": {"C", "c3"},
    ...     "C_c2c3": {"C", "c2", "c3"},
    ...     "D_c4": {"D", "c4"},
    ...     "D_c3": {"D", "c3"},
    ...     "D_c3c4": {"D", "c3", "c4"}
    ... }
    >>> H = HNXHypergraph(edge_dict)
    >>> players = ["A", "B", "C", "D"]
    >>> best_matching = local_search_perfect_matching(H, valuations, players, threshold)
    >>> best_matching == {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
    True

    """
    from collections import deque

    matching: Dict[str, str] = {}  # player -> edge_name
    used_items: Set[str] = set()

    # Check whether the bundle is valid: its value for the player is at least the threshold
    def is_valid_bundle(player: str, bundle: Set[str]) -> bool:
        return sum(valuations[player].get(item, 0) for item in bundle) >= threshold

    def augment_path(player: str, edge: str, parent: Dict[str, Tuple[str, str]]):
        # While the current player is in the alternating tree – walk back toward the root
        while player in parent:
            prev_player, prev_edge = parent[player]
            # Update: assign the current player to the new edge
            matching[player] = edge
            # Move up the tree
            edge = prev_edge
            player = prev_player
        # Finally, assign the root (which has no parent)
        matching[player] = edge
        # Update used items: remove the players from the edge and keep only the items
        used_items.update(set(H.edges[edge]) - set(players))
        logger.debug("Matching after augment: %s", matching)
        logger.debug("Used items after augment: %s", used_items)

    def build_alternating_tree(start_player: str) -> bool:
        queue = deque([start_player]) # Start building an alternating tree from the unmatched player
        # parent: for each player, record the parent player and the connecting edge
        parent: Dict[str, Tuple[str, str]] = {}  # player -> (parent_player, parent_edge)
        visited_players: Set[str] = {start_player} # Track players already visited
        visited_edges: Set[str] = set() # Track edges already checked

        while queue: # Build the alternating tree using BFS – prioritizing minimum swaps
            current_player = queue.popleft()
            for edge_name in H.edges:
                if edge_name in visited_edges:
                    continue # Already visited this edge
                logger.debug("Visiting edge %s", edge_name)

                edge_nodes = set(H.edges[edge_name]) # Get all nodes in the edge
                if current_player not in edge_nodes: # If the arc does not include the current player – not applicable
                    logger.debug(f"Skipping edge {edge_name} – doesn't include {current_player}")
                    continue

                bundle = edge_nodes - {current_player} # Extract the bundle (items only) – remove the current player from the edge
                bundle_items = bundle - set(players)
                logger.debug(f"Checking edge {edge_name} with bundle {bundle_items} for player {current_player}")
                if not is_valid_bundle(current_player, bundle_items): # Skip if the bundle doesn't satisfy the threshold
                    continue

                visited_edges.add(edge_name) # add this edge

                if bundle_items.isdisjoint(used_items): # All items in the bundle are available → augment!
                    augment_path(current_player, edge_name, parent) # check the path
                    return True # We were able to expand the match

                for p, e in matching.items(): #  Otherwise, attempt to swap bundles with other players
                    if not bundle_items.isdisjoint(set(H.edges[e]) - set(players)): # If there is an overlap between the current package and the package of p
                        if p not in visited_players:
                            # Add this player to the tree, with the arrival arc
                            visited_players.add(p)
                            parent[p] = (current_player, edge_name)
                            queue.append(p) # Add to queue

        # If we reached here, we couldn't augment the matching for start_player
        logger.debug("Building alternating tree for player: %s", start_player)
        logger.debug("Used items so far: %s", used_items)
        logger.debug("Parent map: %s", parent)
        return False

    for player in players: # Try to match every player
        if player not in matching:
            success = False
            visited_players = set()
            visited_edges = set()
            while not success: # As long as there is no match for the player, continue.
                success = build_alternating_tree(player) # Connect it to Hyper Edge if possible.
                if not success: # If it is not yet possible
                    edge_to_add = extend_alternating_tree( # We will see if it is possible to expand the tree, that is, to distribute more gifts.
                        H, visited_players, visited_edges, players, valuations, threshold
                    )
                    if edge_to_add is None:
                        break  # Couldn't extend further
                    # If we managed to find an expanding edge – we will add it to the list of arcs that are examined in the next loop.
                    visited_edges.add(edge_to_add)
            if not success:
                for player in players: # run through all the players
                    if player not in matching: # If the players are not yet in a match
                        success = False
                        visited_players = {player}
                        visited_edges = set()
                        while not success:
                            success = build_alternating_tree(player)
                            if not success:
                                edge_to_add = extend_alternating_tree(
                                    H, visited_players, visited_edges, players, valuations, threshold
                                )
                                if edge_to_add is None:
                                    break  # Give up and continue to next player/threshold
                                visited_edges.add(edge_to_add)
                        # No return {} here — just continue

        # Build final allocation
        result: Dict[str, Set[str]] = {}
        used_items: Set[str] = set()

        # Build allocation from matching and track used items
        for player, edge_name in matching.items():
            items = set(H.edges[edge_name]) - {player}
            bundle = items - set(players)
            result[player] = set(bundle)
            used_items |= bundle

        # Fallback: ensure every player gets at least one gift
        for player in players:
            if not result.get(player):
                # choose highest-value remaining item for this player
                candidates = [
                    (valuations[player][item], item)
                    for item in valuations[player]
                    if item not in used_items
                ]
                if candidates:
                    _, pick = max(candidates)
                    result[player] = {pick}
                    used_items.add(pick)

        logger.info("Starting local search for perfect matching")
        logger.debug("Players: %s", players)
        logger.debug("Threshold: %f", threshold)
        logger.info("Finished local search. Final matching: %s", matching)
        logger.debug("Constructed allocation: %s", result)
        return result


if __name__ == "__main__":
    # Run all embedded doctests when executing this script directly
    import doctest
    print("\n", doctest.testmod(), "\n")




"""

GOAL: find maximum T such that
      each child can get value at least T.

Definition: T is feasible == there exists a perfect matching in the hypergraph of T.


LOW =0
HIGH=min[i] sum[vi]

"""
