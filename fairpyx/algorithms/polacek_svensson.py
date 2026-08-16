"""
An implementation of the algorithms in:
"Quasi-Polynomial Local Search for Restricted Max-Min Fair Allocation"
By Lukas Polacek and Ola Svensson (2014)
[https://arxiv.org/pdf/1205.1373]

Programmer: Rotem Melamed
Date: 10/02/2026
"""
from fairpyx.allocations import AllocationBuilder
from fairpyx.instances import Instance
from typing import Set, Dict, List, Optional, Any
import math
import logging


# --- Logging Setup ---
logger = logging.getLogger("QP_Local_Search")


# --- Helper Class ---
class AlternatingTree:
    """
    Alternating tree data structure used in the local search algorithm (Section 3 of the paper).

    The tree is rooted at an unmatched player p0. It maintains two edge sets:
      - A (addable edges): bundles of items that could be assigned to a player in the tree.
        Each A-edge represents a potential new assignment (e_P = player, e_R = items).
      - B (blocking edges): current matching edges that conflict with an A-edge.
        A B-edge (e') blocks an A-edge (e) when they share at least one item.

    Edges are classified by distance from the root:
      - Fat edges (distance 0): a single item whose value >= T/alpha for the player.
      - Thin edges (distance 1): a bundle of multiple small items summing to >= T/alpha.

    The tree grows by adding A-edges and their blockers, and collapses when a free
    A-edge (one with no blockers) is found, allowing augmentation of the matching.
    """
    def __init__(self, root_player: Any, T: float, alpha: float):
        """
        Initialize an alternating tree rooted at an unmatched player.

        Args:
            root_player: The unmatched player p0 that roots this tree.
            T: The target value being tested in binary search.
            alpha: The approximation factor (4 + epsilon). The effective threshold
                   for each bundle is T / alpha.
        """
        self.root = root_player
        self.T = T
        self.alpha = alpha

        # Key: tuple(sorted items) -> edge metadata dict
        # metadata keys: items, player, distance, type, layer, parent
        # parent -> key of the preceding edge on the alternating path (or None for root)
        self.A_edges: Dict[tuple, dict] = {}
        self.B_edges: Dict[tuple, dict] = {}

        self.visited_bundles = set()

    def get_distance(self, bundle_items: tuple, player: Any, instance: Instance) -> int:
        """
        Classify an edge as fat (distance 0) or thin (distance 1).

        Per the paper's definition:
          - A fat edge contains a single item j with v_{ij} >= T/alpha.
          - A thin edge is a bundle of items whose total value >= T/alpha,
            but no single item alone meets the threshold.

        Args:
            bundle_items: The items in the bundle.
            player: The player this edge is for.
            instance: Problem instance for value lookups.

        Returns:
            0 for fat edges, 1 for thin edges.

        >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 2, "z": 3}})
        >>> tree = AlternatingTree("Alice", T=20, alpha=4.1)
        >>> tree.get_distance(("x",), "Alice", instance)  # 10 >= 20/4.1 ≈ 4.88 -> fat
        0
        >>> tree.get_distance(("y", "z"), "Alice", instance)  # multi-item -> thin
        1
        >>> tree.get_distance(("y",), "Alice", instance)  # 2 < 4.88 -> thin
        1
        """
        threshold = self.T / self.alpha
        if len(bundle_items) == 1:
            val = instance.agent_item_value(player, bundle_items[0])
            if val >= threshold:
                return 0  # Fat edge
        return 1  # Thin edge

    def _find_greedy_bundle(self, agent: Any, instance: Instance, excluded_items: set) -> Optional[tuple]:
        """
        Greedily construct a bundle C for player i such that v_i(C) >= T/alpha.

        This is the purely combinatorial approach from the paper — when the algorithm
        says "let e be an addable edge", we greedily build such a bundle on-the-fly
        rather than enumerating all possible bundles.

        Strategy:
          1. First check for fat edges: any single item with value >= T/alpha.
          2. If none, sort available items by descending value and greedily collect
             until the threshold is met (thin edge).

        Args:
            agent: The player to build a bundle for.
            instance: Problem instance for value lookups.
            excluded_items: Items already used in the tree (A-edges and B-edges),
                            which cannot appear in a new addable edge.

        Returns:
            A tuple of items forming a valid bundle, or None if the threshold
            cannot be reached with the remaining available items.

        >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 3, "z": 2}})
        >>> tree = AlternatingTree("Alice", T=20, alpha=4.1)  # threshold ≈ 4.88

        Fat edge found (single item x=10 >= 4.88):
        >>> tree._find_greedy_bundle("Alice", instance, set())
        ('x',)

        With x excluded, greedy collects y+z = 5 >= 4.88 (thin edge):
        >>> tree._find_greedy_bundle("Alice", instance, {"x"})
        ('y', 'z')

        All items excluded -> None:
        >>> tree._find_greedy_bundle("Alice", instance, {"x", "y", "z"})

        Agent values nothing -> None:
        >>> instance2 = Instance(valuations={"Alice": {"x": 0, "y": 0}})
        >>> tree2 = AlternatingTree("Alice", T=20, alpha=4.1)
        >>> tree2._find_greedy_bundle("Alice", instance2, set())
        """
        threshold = self.T / self.alpha

        # Collect available items: agent values them > 0 and they're not in the tree
        available = []
        for item in instance.items:
            if item in excluded_items:
                continue
            val = instance.agent_item_value(agent, item)
            if val > 0:
                available.append((item, val))

        # 1. Check for fat edges first (single item >= threshold, distance 0)
        for item, val in available:
            if val >= threshold:
                bundle = (item,)
                if bundle not in self.visited_bundles:
                    logger.debug(f"    Greedy: found fat edge ({item}) for {agent}, value={val:.2f} >= threshold={threshold:.2f}")
                    return bundle

        # 2. Greedily collect items by descending value for a thin edge (distance 1)
        available.sort(key=lambda x: x[1], reverse=True)
        bundle_items = []
        bundle_value = 0.0
        for item, val in available:
            bundle_items.append(item)
            bundle_value += val
            if bundle_value >= threshold:
                bundle = tuple(sorted(bundle_items))
                if bundle not in self.visited_bundles:
                    logger.debug(f"    Greedy: found thin edge {bundle} for {agent}, value={bundle_value:.2f} >= threshold={threshold:.2f}")
                    return bundle
                return None

        logger.debug(f"    Greedy: cannot reach threshold={threshold:.2f} for {agent}, max reachable={bundle_value:.2f}")
        return None

    def find_addable_edge(self, instance: Instance, max_distance: int) -> Optional[dict]:
        """
        Find an addable edge e of minimum distance from the root (Algorithm 1, line 3).

        An edge e = (player, items) is addable if:
          - player is the root or was brought into the tree via a B-edge
          - items are disjoint from all items currently in the tree
          - the bundle has not been visited before
          - total distance (parent distance + edge cost) <= max_distance

        The method prefers edges for the root player (they collapse directly to a
        successful augmentation) and then edges with minimum total distance.

        Args:
            instance: Problem instance for value lookups.
            max_distance: Maximum allowed distance from root, which is
                          2 * ceil(log_{1+eps/3}(|P|)) + 1 per the paper.

        Returns:
            Edge metadata dict with keys {items, player, distance, type, layer, parent},
            or None if no addable edge exists within the distance bound.

        >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 5}})
        >>> tree = AlternatingTree("Alice", T=20, alpha=4.1)  # threshold ≈ 4.88

        Root player can get a fat edge (x=10 >= 4.88):
        >>> edge = tree.find_addable_edge(instance, max_distance=5)
        >>> edge['player'], edge['items'], edge['type']
        ('Alice', ('x',), 'fat')

        No edge if max_distance is too small and only thin edges exist:
        >>> tree2 = AlternatingTree("Alice", T=100, alpha=4.1)  # threshold ≈ 24.4
        >>> tree2.find_addable_edge(instance, max_distance=0)
        """
        best_edge = None
        min_dist = float('inf')
        best_is_root = False

        # Candidate players: root + all players brought in via B-edges (their matching was blocked)
        candidate_players = {self.root}
        for meta in self.B_edges.values():
            candidate_players.add(meta['player'])

        # Collect all items currently in the tree (cannot be reused in a new A-edge)
        tree_items = set()
        for meta in self.A_edges.values():
            tree_items.update(meta['items'])
        for meta in self.B_edges.values():
            tree_items.update(meta['items'])

        logger.debug(f"  [find_addable_edge] candidates={len(candidate_players)}, tree_items={len(tree_items)}, max_dist={max_distance}")

        for player in candidate_players:
            # Determine parent distance (root has distance 0)
            parent_dist = 0
            if player != self.root:
                found = False
                for b_meta in self.B_edges.values():
                    if b_meta['player'] == player:
                        parent_dist = b_meta['distance']
                        found = True
                        break
                if not found:
                    continue

            if parent_dist >= max_distance:
                continue

            # Greedily construct ONE bundle for this player
            bundle = self._find_greedy_bundle(player, instance, tree_items)
            if bundle is None:
                continue

            edge_cost = self.get_distance(bundle, player, instance)
            total_dist = parent_dist + edge_cost

            # Check if this edge is addable and better than the best found so far
            if total_dist <= max_distance:
                is_root = (player == self.root)
                if (is_root and not best_is_root) or \
                   (is_root == best_is_root and total_dist < min_dist):
                    min_dist = total_dist
                    best_is_root = is_root
                    if edge_cost == 0:
                        edge_type = 'fat'
                        layer = 2 * (total_dist // 2)
                    else:
                        edge_type = 'thin'
                        layer = 2 * (total_dist // 2) + 1
                    best_edge = {
                        'items': bundle,
                        'player': player,
                        'distance': total_dist,
                        'type': edge_type,
                        'layer': layer,
                        'parent': None
                    }
        return best_edge

    def get_blocking_edges(self, edge_to_add: dict, matching: Dict[Any, Set]) -> List[dict]:
        """
        Find all matching edges that block a given A-edge (Algorithm 1, line 5).

        An edge e' in the current matching M blocks the addable edge e if they
        share at least one item (e_R intersect e'_R != empty). When this happens,
        the player holding that matching edge is "brought into" the tree via a B-edge,
        and may later get a new A-edge (a replacement bundle).

        Args:
            edge_to_add: The A-edge being added to the tree.
            matching: Current matching M, mapping player -> set of assigned items.

        Returns:
            List of B-edge metadata dicts for all blocking matching edges.

        >>> tree = AlternatingTree("Alice", T=20, alpha=4.1)
        >>> edge = {'items': ('x', 'y'), 'distance': 1, 'layer': 1}

        Bob holds x -> blocks the edge:
        >>> blockers = tree.get_blocking_edges(edge, {"Bob": {"x", "z"}})
        >>> len(blockers)
        1
        >>> blockers[0]['player']
        'Bob'

        Carol holds w -> no overlap, no blocking:
        >>> tree.get_blocking_edges(edge, {"Carol": {"w"}})
        []
        """
        blockers = []
        new_items = set(edge_to_add['items'])
        add_layer = edge_to_add.get('layer', edge_to_add['distance'])

        for m_player, m_bundle in matching.items():
            if not m_bundle:
                continue
            if new_items.isdisjoint(m_bundle):
                continue

            b_items = tuple(sorted(m_bundle))

            blocker = {
                'items': b_items,
                'player': m_player,
                'distance': edge_to_add['distance'],
                'layer': add_layer,
                'blocking_who': edge_to_add['items'],
                'parent': tuple(sorted(edge_to_add['items'])),
            }
            blockers.append(blocker)

        return blockers


    def prune(self, max_allowed_distance: int):
        """
        Remove all edges with distance greater than max_allowed_distance (Algorithm 1, line 16).

        After a partial collapse, the paper says: "Remove all edges in A of greater
        distance than e' and the edges in B that blocked these edges." This prevents
        the tree from growing unboundedly and is key to the quasi-polynomial time bound.

        Args:
            max_allowed_distance: The distance of the last removed B-edge (e').
                                  All edges with distance > this value are removed.

        >>> tree = AlternatingTree("Alice", T=20, alpha=4.1)
        >>> tree.A_edges[('x',)] = {'items': ('x',), 'distance': 1}
        >>> tree.A_edges[('y',)] = {'items': ('y',), 'distance': 3}
        >>> tree.visited_bundles = {('x',), ('y',)}
        >>> tree.B_edges[('z',)] = {'items': ('z',), 'distance': 3}
        >>> tree.prune(2)
        >>> ('x',) in tree.A_edges  # distance 1 <= 2, kept
        True
        >>> ('y',) in tree.A_edges  # distance 3 > 2, removed
        False
        >>> ('z',) in tree.B_edges  # distance 3 > 2, removed
        False
        >>> ('y',) in tree.visited_bundles  # also removed from visited
        False
        """
        keys_to_remove_A = [k for k, v in self.A_edges.items() if v['distance'] > max_allowed_distance]
        for k in keys_to_remove_A:
            del self.A_edges[k]
            self.visited_bundles.discard(k)

        keys_to_remove_B = [k for k, v in self.B_edges.items() if v['distance'] > max_allowed_distance]
        for k in keys_to_remove_B:
            del self.B_edges[k]

        logger.info(f"    [Algorithm 1, line 16] Pruning edges with distance > {max_allowed_distance}: "
                     f"removed {len(keys_to_remove_A)} A-edges, {len(keys_to_remove_B)} B-edges")


def is_satisfied(alloc: AllocationBuilder, agent, threshold: float) -> bool:
    """Check if an agent's bundle value meets or exceeds the threshold."""
    return alloc.instance.agent_bundle_value(agent, alloc.bundles.get(agent, [])) >= threshold


def safe_swap(alloc: AllocationBuilder, player_losing: Any, items_losing, player_getting: Any, items_getting):
    """
    Transfer items between players in the matching.

    Used during the collapse procedure when M <- M \\ {e'} U {e}:
    the player loses their old matching edge (e') and gains the new A-edge (e).

    Args:
        alloc: The allocation builder tracking the current matching.
        player_losing: The player whose items are being removed.
        items_losing: The items to remove from this player's bundle.
        player_getting: The player receiving new items (same player in a swap).
        items_getting: The items to assign to this player.

    >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 5, "z": 8}})
    >>> alloc = AllocationBuilder(instance)
    >>> alloc.give("Alice", "x")
    >>> alloc.give("Alice", "y")
    >>> sorted(alloc.bundles["Alice"])
    ['x', 'y']
    >>> safe_swap(alloc, "Alice", ("x", "y"), "Alice", ("z",))
    >>> sorted(alloc.bundles["Alice"])
    ['z']
    """
    for item in items_losing:
        if item in alloc.bundles[player_losing]:
            alloc.bundles[player_losing].remove(item)
        if hasattr(alloc, 'remaining_item_capacities'):
            alloc.remaining_item_capacities[item] = alloc.remaining_item_capacities.get(item, 0) + 1
        if hasattr(alloc, 'remaining_conflicts'):
            alloc.remaining_conflicts.discard((player_losing, item))
            for conflicting_item in alloc.instance.item_conflicts(item):
                alloc.remaining_conflicts.discard((player_losing, conflicting_item))

    for item in items_getting:
        alloc.give(player_getting, item)


# --- Algorithm 1: Single augmentation step ---
def algorithm1_augment(alloc: AllocationBuilder, T: float, epsilon: float = 0.1) -> bool:
    """
    Algorithm 1 from the paper: augment a partial matching by one player.

    Input:  A partial matching M (stored in alloc)
    Output: A matching of increased size (one more player satisfied),
            assuming T <= T_OPT.

    Finds an unmatched player p0, makes it the root of an alternating tree,
    and grows the tree by finding addable edges and their blockers. When a
    free addable edge is found (no blockers), the collapse procedure walks
    back along the alternating path, swapping items, until the root player
    gets assigned — increasing the matching size by one.

    Args:
        alloc: AllocationBuilder with the current partial matching M.
        T: Target value. Each player must achieve value >= T / (4 + epsilon).
        epsilon: Approximation parameter.

    Returns:
        True if the matching was augmented (one more player now satisfied).
        False if no augmenting path exists (T_OPT < T, this T is infeasible).

    >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 20, "z": 15}, "Bob": {"x": 10, "y": 20, "z": 15}})
    >>> alloc = AllocationBuilder(instance)
    >>> algorithm1_augment(alloc, T=20, epsilon=0.1)  # augments one player
    True
    """
    instance = alloc.instance
    alpha = 4 + epsilon
    threshold = T / alpha

    # Find an unmatched (unsatisfied) player p0
    unsatisfied_agents = [p for p in instance.agents if not is_satisfied(alloc, p, threshold)]
    if not unsatisfied_agents:
        logger.info(f"[Algorithm 1] All players already satisfied at threshold {threshold:.2f}")
        return True

    root_agent = unsatisfied_agents[0]
    logger.info(f"\n[Algorithm 1, line 1] Find unmatched player p0={root_agent}, make it root of alternating tree")

    alt_tree = AlternatingTree(root_agent, T, alpha)

    # Maximum distance bound: 2 * ceil(log_{1+eps/3}(|P|)) + 1
    num_players = len(instance.agents)
    base = (alpha - 1) / 3
    max_dist = 2 * math.ceil(math.log(num_players, base)) + 1 if num_players > 0 else 1
    logger.info(f"[Algorithm 1, line 2] Max distance bound: 2*ceil(log_{{{base:.3f}}}({num_players})) + 1 = {max_dist}")

    # while there is an addable edge within distance bound
    while True:
        # Algorithm 1, line 3: "Find an addable edge e of minimum distance from the root"
        logger.info(f"[Algorithm 1, line 3] Searching for addable edge within distance {max_dist}...")
        addable_edge = alt_tree.find_addable_edge(instance, max_dist)

        if addable_edge is None:
            logger.info(f"[Algorithm 1, line 2] No addable edge within distance {max_dist} -> exit while loop")
            break

        e_key = tuple(sorted(addable_edge['items']))
        logger.info(f"[Algorithm 1, line 3] Found addable edge e: player={addable_edge['player']}, "
                     f"items={e_key}, type={addable_edge['type']}, distance={addable_edge['distance']}")

        # Algorithm 1, line 4: "A <- A U {e}"
        parent_key = None
        for k, b in alt_tree.B_edges.items(): # k is the key of the B-edge, b is the metadata dict
            if b['player'] == addable_edge['player']: # if the B-edge brings the same player into the tree, it is the parent of this A-edge
                parent_key = k
                break
        addable_edge['parent'] = parent_key

        alt_tree.A_edges[e_key] = addable_edge
        alt_tree.visited_bundles.add(e_key)
        logger.info(f"[Algorithm 1, line 4] A <- A U {{e}}, |A|={len(alt_tree.A_edges)}")

        # Check for blocking edges from the current matching
        current_matching = {a: s for a, s in alloc.bundles.items() if s}
        blocking_edges = alt_tree.get_blocking_edges(addable_edge, current_matching)

        if blocking_edges:
            # Algorithm 1, line 5-6: "if e has blocking edges b1,...,bk then B <- B U {b1,...,bk}"
            for b in blocking_edges:
                b_key = tuple(sorted(b['items']))
                alt_tree.B_edges[b_key] = b
                logger.info(f"[Algorithm 1, line 6] B <- B U {{b}}: player={b['player']}, items={b_key} blocks {e_key}")
            logger.info(f"[Algorithm 1, line 6] Added {len(blocking_edges)} blocking edge(s), |B|={len(alt_tree.B_edges)}")

        else:
            # Algorithm 1, line 7: "else (collapse procedure)"
            logger.info(f"[Algorithm 1, line 7] Edge {e_key} has no blocking edges -> begin COLLAPSE procedure")

            edge_player = addable_edge['player']
            if edge_player != root_agent:
                has_blocker = any(b['player'] == edge_player for b in alt_tree.B_edges.values()) # check for a blocker that brought this player into the tree
                if not has_blocker:
                    logger.warning(f"[Algorithm 1] Collapse skipped: player {edge_player} not connected to tree via B-edge")
                    continue

            curr_key = e_key
            last_removed_blocker = None
            collapse_succeeded = False

            # Algorithm 1, line 8: "while e has no blocking edges do"
            while True:
                curr_e = alt_tree.A_edges[curr_key]

                # Check if any B-edge still blocks the current A-edge
                has_blocker = any(
                    not set(curr_e['items']).isdisjoint(b_edge['items'])
                    for b_edge in alt_tree.B_edges.values()
                )
                if has_blocker: # if there is still a blocker, we cannot collapse this edge
                    logger.info(f"[Algorithm 1, line 8] Edge {curr_key} is now blocked -> exit collapse while loop")
                    break

                player = curr_e['player']

                # Algorithm 1, line 13: "else M <- M U {e}, return M"
                if player == root_agent:
                    logger.info(f"[Algorithm 1, line 13] Reached root player {player}: M <- M U {{e}}, assigning items {curr_e['items']}")
                    for item in curr_e['items']:
                        alloc.give(player, item)
                    collapse_succeeded = True
                    break

                # Algorithm 1, line 9-10: "if there is e' in B such that e'_P = e_P then"
                #   "M <- M \ {e'} U {e}"
                e_prime_key = None
                e_prime = None
                for k, b in alt_tree.B_edges.items():
                    if b['player'] == player:
                        e_prime_key, e_prime = k, b
                        break

                if e_prime is None:
                    logger.warning(f"[Algorithm 1] Collapse: no B-edge found for player {player}")
                    break

                # Algorithm 1, line 10: "M <- M \ {e'} U {e}"
                logger.info(f"[Algorithm 1, line 10] M <- M \\ {{e'}} U {{e}}: "
                             f"player {player} swaps {e_prime['items']} -> {curr_e['items']}")
                safe_swap(alloc, player, e_prime['items'], player, curr_e['items'])

                # Algorithm 1, line 11: "A <- A \ {e}, B <- B \ {e'}"
                del alt_tree.A_edges[curr_key]
                del alt_tree.B_edges[e_prime_key]
                last_removed_blocker = e_prime
                logger.info(f"[Algorithm 1, line 11] A <- A \\ {{e}}, B <- B \\ {{e'}}, |A|={len(alt_tree.A_edges)}, |B|={len(alt_tree.B_edges)}")

                # Algorithm 1, line 12: "Let e'' in A be the edge that e' was blocking, e <- e''"
                parent_key = e_prime.get('parent')
                if parent_key is None or parent_key not in alt_tree.A_edges:
                    logger.info(f"[Algorithm 1, line 12] No parent A-edge for removed blocker -> collapse ends")
                    break
                logger.info(f"[Algorithm 1, line 12] e <- e'' (the edge e' was blocking): {parent_key}")
                curr_key = parent_key

            if collapse_succeeded:
                # Algorithm 1, line 14: "return M" (matching of increased size)
                logger.info(f"[Algorithm 1, line 14] Collapse SUCCEEDED: matching augmented for root {root_agent}")
                return True
            else:
                # Algorithm 1, line 15-16: Prune edges with distance > last removed blocker
                if last_removed_blocker:
                    dist = last_removed_blocker['distance']
                    logger.info(f"[Algorithm 1, line 15-16] Partial collapse ended. Pruning to distance {dist}")
                    alt_tree.prune(dist)
                else:
                    logger.warning(f"[Algorithm 1] Collapse failed with no blocker removed")

    # Algorithm 1, line 18: "return T_OPT is less than T"
    logger.info(f"[Algorithm 1, line 18] No augmenting path for {root_agent} -> T_OPT < T, this T={T:.2f} is infeasible")
    return False


# --- Outer loop: repeatedly call Algorithm 1 to grow the matching ---
def qp_local_search(alloc: AllocationBuilder, T: float, epsilon: float = 0.1) -> bool:
    """
    Repeatedly call Algorithm 1 to grow the partial matching until all players
    are satisfied or the target T is determined to be infeasible.

    Each call to Algorithm 1 takes the current partial matching M and either
    augments it by one (one more player satisfied) or returns that T is too high.
    This loop continues until every player has value >= T / (4 + epsilon).

    Args:
        alloc: AllocationBuilder with the current (possibly empty) matching.
        T: Target value. Each player must achieve value >= T / (4 + epsilon).
        epsilon: Approximation parameter. Smaller epsilon gives a better
                 approximation ratio 1/(4+epsilon) but increases running time.

    Returns:
        True if all players are satisfied (value >= T/alpha), False if the
        algorithm gets stuck (no addable edge found), meaning T is too high.

    >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 20, "z": 15}, "Bob": {"x": 10, "y": 20, "z": 15}})
    >>> alloc = AllocationBuilder(instance)
    >>> qp_local_search(alloc, T=20, epsilon=0.1)  # threshold = 20/4.1 ≈ 4.88
    True
    >>> all(len(alloc.bundles[a]) > 0 for a in instance.agents)
    True

    T too high -> infeasible:
    >>> alloc2 = AllocationBuilder(instance)
    >>> qp_local_search(alloc2, T=1000, epsilon=0.1)  # threshold ≈ 243.9, impossible
    False
    """
    instance = alloc.instance
    alpha = 4 + epsilon
    threshold = T / alpha

    logger.info(f"[Outer Loop] Starting: T={T:.2f}, threshold=T/alpha={threshold:.2f}, alpha={alpha:.2f}")

    while any(not is_satisfied(alloc, p, threshold) for p in instance.agents):
        # Call Algorithm 1: augment the matching by one player
        success = algorithm1_augment(alloc, T, epsilon)
        if not success:
            return False
        unsatisfied_count = sum(1 for p in instance.agents if not is_satisfied(alloc, p, threshold))
        logger.info(f"[Outer Loop] Remaining unsatisfied: {unsatisfied_count}")

    logger.info(f"[Outer Loop] ALL PLAYERS SATISFIED at threshold T/alpha = {threshold:.2f}")
    return True


def qp_max_min_allocation(alloc: AllocationBuilder, epsilon: float = 0.1):
    """
    Find the maximum T such that all agents can achieve value >= T/(4+epsilon).

    Uses binary search over T, running Algorithm 1 (the purely combinatorial local
    search) at each step. The local search's success or failure determines whether
    T is feasible.

    The approximation ratio is 1/(4+epsilon), matching the guarantee from the paper.
    The binary search converges when T_high - T_low < tolerance (1e-3).

    Args:
        alloc: an AllocationBuilder, which tracks the allocation and the remaining
               capacity for items and agents.
        epsilon: Approximation parameter (default 0.1 for 1/4.1-approximation).
                 Smaller values give tighter guarantees but slower runtime.

    >>> from fairpyx.adaptors import divide
    >>> from fairpyx.utils.test_utils import stringify

    >>> instance = Instance(valuations={"Alice": {"x": 10, "y": 20, "z": 15}, "Bob": {"x": 10, "y": 20, "z": 15}})
    >>> result = divide(qp_max_min_allocation, instance=instance, epsilon=0.1)
    >>> set(result.keys()) == {"Alice", "Bob"}
    True
    >>> all(len(bundle) > 0 for bundle in result.values())
    True

    Each agent's value should meet the approximation guarantee:
    >>> alpha = 4.1
    >>> all(sum(instance.agent_item_value(a, i) for i in b) >= alpha for a, b in result.items()) #
    True

    Zero-value instance returns empty bundles:
    >>> instance0 = Instance(valuations={"Alice": {"x": 0}, "Bob": {"x": 0}})
    >>> divide(qp_max_min_allocation, instance=instance0)
    {'Alice': [], 'Bob': []}
    """
    instance = alloc.instance
    logger.info(f"\n{'='*60}")
    logger.info(f"QUASI-POLYNOMIAL MAX-MIN ALLOCATION")
    logger.info(f"Agents: {list(instance.agents)}, Items: {list(instance.items)}")
    logger.info(f"Epsilon: {epsilon}, Approximation ratio: 1/{4+epsilon:.2f}")
    logger.info(f"{'='*60}\n")

    # Determine search range for T: [0, max total value any agent can achieve]
    max_total_value = 0
    for agent in instance.agents:
        total_value = sum(instance.agent_item_value(agent, item) for item in instance.items)
        max_total_value = max(max_total_value, total_value)

    if max_total_value == 0:
        logger.warning("All items have zero value for all agents")
        return

    T_low = 0.0

    # The paper's analysis shows that the optimal max-min value T_OPT is at most 4 times the maximum value any single agent can get from all items (since the approximation ratio is 1/(4+epsilon)). We add a small epsilon buffer to ensure we search above the true T_OPT.
    T_high = max_total_value *(4 + epsilon)
    best_T = 0.0
    best_allocation = {agent: set() for agent in instance.agents}
    tolerance = 1e-3

    iteration = 0
    max_iterations = int(math.log2(T_high / tolerance)) + 10
    logger.info(f"[Binary Search] Range: [{T_low}, {T_high}], tolerance={tolerance}, max_iterations={max_iterations}")

    while T_high - T_low > tolerance and iteration < max_iterations:
        T_mid = (T_low + T_high) / 2
        iteration += 1

        logger.info(f"\n[Binary Search] Iteration {iteration}: testing T = {T_mid:.4f} (range [{T_low:.4f}, {T_high:.4f}])")

        # Start with a fresh (empty) matching for each T — edge classifications (fat/thin)
        # depend on T, so reusing a matching from a different T leads to incorrect classifications.
        search_alloc = AllocationBuilder(instance)
        success = qp_local_search(search_alloc, T_mid, epsilon)

        if success:
            best_T = T_mid
            best_allocation = search_alloc.sorted()
            T_low = T_mid
            logger.info(f"[Binary Search] T={T_mid:.4f} is FEASIBLE -> search higher (T_low <- {T_mid:.4f})")
        else:
            T_high = T_mid
            logger.info(f"[Binary Search] T={T_mid:.4f} is INFEASIBLE -> search lower (T_high <- {T_mid:.4f})")

    # Final result
    logger.info(f"\n{'='*60}")
    logger.info(f"[Binary Search] COMPLETE after {iteration} iterations")
    logger.info(f"[Binary Search] Best T found: {best_T:.4f}, threshold: {best_T/(4+epsilon):.4f}")
    logger.info(f"{'='*60}")

    if best_T == 0.0:
        logger.warning("No feasible allocation found")
        return

    # Post-processing: assign remaining unallocated items greedily.
    # The core algorithm only allocates enough items to meet the threshold,
    # so leftover items are assigned to the agent with the lowest bundle value
    # (among agents who actually want the item, i.e. value > 0).
    allocated_items = set()
    for bundle in best_allocation.values():
        allocated_items.update(bundle)
    remaining_items = set(instance.items) - allocated_items

    if remaining_items:
        logger.info(f"\n[Post-processing] {len(remaining_items)} unallocated items: {remaining_items}")
        bundle_values = {
            agent: sum(instance.agent_item_value(agent, item) for item in best_allocation.get(agent, []))
            for agent in instance.agents
        }
        for item in remaining_items:
            # Only consider agents who want this item (value > 0)
            candidates = [a for a in instance.agents if instance.agent_item_value(a, item) > 0]
            if not candidates:
                continue
            # Assign to the candidate with the lowest current bundle value
            best_agent = min(candidates, key=lambda a: bundle_values[a])
            best_allocation[best_agent].append(item)
            bundle_values[best_agent] += instance.agent_item_value(best_agent, item)
            logger.info(f"  Assigned {item} to {best_agent} (bundle_value now={bundle_values[best_agent]:.4f})")

    # Apply the final allocation to the AllocationBuilder
    for agent, bundle in best_allocation.items():
        for item in bundle:
            alloc.give(agent, item)

    logger.info(f"\nFinal allocation:")
    threshold = best_T / (4 + epsilon)
    for agent, bundle in best_allocation.items():
        value = sum(instance.agent_item_value(agent, item) for item in bundle)
        logger.info(f"  {agent}: items={bundle}, value={value:.4f} (threshold={threshold:.4f})")


if __name__ == "__main__":
    from fairpyx.adaptors import divide

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    instance = Instance(
        valuations={
            "Alice": {"item1": 10, "item2": 20, "item3": 30},
            "Bob":   {"item1": 10, "item2": 0, "item3": 30},
        }
    )
    allocation = divide(qp_max_min_allocation, instance=instance)
    print(allocation)
