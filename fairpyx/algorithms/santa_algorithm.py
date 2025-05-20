"""
An implementation of the algorithms in:
"Santa Claus Meets Hypergraph Matchings",
by ARASH ASADPOUR - New York University, URIEL FEIGE - The Weizmann Institute, AMIN SABERI - Stanford University,
https://dl.acm.org/doi/abs/10.1145/2229163.2229168
Programmers: May Rozen
Date: 2025-04-23
"""
 # יש במאמר אלגוריתם אחד והוא בנוי באופן מודולרי. כמו כן, גם בספריית fairpyx – לאלגוריתמי חלוקה הוגנת, האלגוריתמים בנויים כך.
 # לכן, גם כאן בניתי את כותרות האלגוריתם באופן כזה.

import numpy as np
from typing import Dict, List, Set, Tuple
from hypernetx import Hypergraph as HNXHypergraph
from fairpyx import Instance, AllocationBuilder
from fairpyx import validate_allocation
import logging
from itertools import chain, combinations

# הגדרת הלוגר
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_allocation_strings(allocation: Dict[str, str]) -> Dict[str, List[Set[str]]]:
    """
    מקבלת הקצאה בפורמט של מחרוזות כמו '1.0*{'c1', 'c3'}'
    ומחזירה: {'Alice': [{'c1', 'c3'}], ...}
    """
    import ast
    parsed = {}
    for agent, bundle_str in allocation.items():
        if "*{" in bundle_str:
            try:
                bundle_part = bundle_str.split("*", 1)[1]
                bundle = ast.literal_eval(bundle_part)
                parsed[agent] = [bundle]
            except Exception:
                parsed[agent] = []
        else:
            parsed[agent] = []
    return parsed

def santa_claus_main(allocation_builder: AllocationBuilder) -> Dict[str, Set[str]]:
    """
    >>> # Test 1: Simple case with 2 players and 3 items
    >>> instance = Instance(
    ...     valuations={"Alice": {"c1": 5, "c2": 0, "c3": 6}, "Bob": {"c1": 0, "c2": 8, "c3": 0}},
    ...     agent_capacities={"Alice": 2, "Bob": 1},
    ...     item_capacities={"c1": 5, "c2": 8, "c3": 6},
    ... )
    >>> allocation_builder = AllocationBuilder(instance=instance)
    >>> result = santa_claus_main(allocation_builder)
    >>> result == {'Alice': {'c1', 'c3'}, 'Bob': {'c2'}}
    True

    >>> # Test 2: More complex case with 4 players and 4 items
    >>> instance = Instance(
    ...     valuations={"A": {"c1": 10, "c2": 0, "c3": 0, "c4": 6}, "B": {"c1": 10, "c2": 8, "c3": 0, "c4": 0}, "C": {"c1": 10, "c2": 8, "c3": 0, "c4": 0}, "D": {"c1": 0, "c2": 0, "c3": 6, "c4": 6}},
    ...     agent_capacities={"A": 1, "B": 1, "C": 1, "D": 1},
    ...     item_capacities={"c1": 1, "c2": 1, "c3": 1, "c4": 1},
    ... )
    >>> allocation_builder = AllocationBuilder(instance=instance)
    >>> result = santa_claus_main(allocation_builder)
    >>> result == {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
    True
    """
    instance = allocation_builder.instance
    agent_names = list(instance.agents)
    item_names = list(instance.items)
    valuations = {
        agent: {
            item: instance.agent_item_value(agent, item)
            for item in item_names
        }
        for agent in agent_names
    }

    all_item_values = [v for val in valuations.values() for v in val.values()]
    high = max(all_item_values) if all_item_values else 0
    low = 0
    best_matching: Dict[str, Set[str]] = {}

    while high - low > 1e-4:
        mid = (low + high) / 2
        if is_threshold_feasible(valuations, mid):
            low = mid
            raw_allocation = solve_configuration_lp(valuations, mid)
            allocation = parse_allocation_strings(raw_allocation)
            fat_items, thin_items = classify_items(valuations, mid)
            H = build_hypergraph(valuations, allocation, fat_items, thin_items, mid)
            best_matching = local_search_perfect_matching(H, valuations, agent_names, threshold=mid)
        else:
            high = mid

    final_allocation = {agent: list(items) for agent, items in best_matching.items()}
    validate_allocation(instance, final_allocation)
    return {agent: set(items) for agent, items in final_allocation.items()}

def is_threshold_feasible(valuations: Dict[str, Dict[str, float]], threshold: float) -> bool:
    """

    בודקת האם קיים שיבוץ שבו כל שחקן מקבל חבילה שערכה לפחות הסף הנתון (threshold).

    הסבר:
    זהו שלב 1 של האלגוריתם – בדיקת סף (Threshold Selection).
    אנו בוחרים ערך סף t, ומנסים לבדוק האם קיימת הקצאה שבה כל שחקן מקבל חבילה שערכה לפחות t.
    לאחר מכן, האלגוריתם מבצע חיפוש בינארי על t, כדי למצוא את הערך המרבי האפשרי.
    הפונקציה הזו עוזרת לקבוע האם עבור ערך מסוים של t קיימת הקצאה חוקית שמספקת כל שחקן.

    Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "Alice": {"c1": 7, "c2": 0, "c3": 4},
    ...     "Bob":   {"c1": 0,  "c2": 8, "c3": 0}
    ... }
    >>> is_threshold_feasible(valuations, 15)
    False
    >>> is_threshold_feasible(valuations, 10)
    False
    >>> is_threshold_feasible(valuations, 8)
    True

    Example 2: 2 Players, 2 Items (conflict)
    >>> valuations = {
    ...     "Alice": {"c1": 10, "c2": 0},
    ...     "Bob":   {"c1": 0, "c2": 9}
    ... }
    >>> is_threshold_feasible(valuations, 10)
    False
    >>> is_threshold_feasible(valuations, 9)
    True
    """
    logger.debug("Starting threshold feasibility check for threshold: %f", threshold)

    for player, items in valuations.items():
        total_value = sum(value for value in items.values())
        logger.debug("Player %s has total value: %f", player, total_value)

        if total_value < threshold:
            logger.warning("Player %s's total value %f is below threshold %f", player, total_value, threshold)
            return False

    logger.info("Threshold feasibility check passed, all players can receive at least %f value", threshold)
    return True

def generate_all_subsets(items: List[str]) -> List[Set[str]]:
    """
    מחזירה את כל תתי-הקבוצות האפשריות (ללא הקבוצה הריקה).
    """
    return [set(comb) for r in range(1, len(items)+1) for comb in combinations(items, r)]

def solve_configuration_lp(valuations: Dict[str, Dict[str, float]], threshold: float) -> Dict[str, str]:
    """
    פונקציה זו פותרת את הבעיה הליניארית (LP) של הקונפיגורציה ומחזירה הקצאה שברית של חבילות לשחקנים.

    הסבר:
    זהו שלב 2 של האלגוריתם – פתרון תכנות ליניארי (Configuration LP Relaxation).
    נגדיר משתנים בינאריים x_{i,S} שמציינים אם שחקן i מקבל חבילה S ⊆ R שערכה הכולל לפחות t.
    הפונקציה הזו פותרת את הבעיה הליניארית בצורת Relaxation ומחזירה עבור כל שחקן אוסף של חבילות אפשריות שערכן לפחות t.
    זוהי הקצאה חלקית ולא בהכרח שלמה, אך נשתמש בה בהמשך כדי לבנות את ההיפרגרף.

    Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "Alice": {"c1": 7, "c2": 0, "c3": 8},
    ...     "Bob":   {"c1": 0,  "c2": 8, "c3": 0}
    ... }
    >>> solve_configuration_lp(valuations, 8)
    {'Alice': "1.0*{'c1', 'c3'}", 'Bob': "1.0*{'c2'}"}

    """
    allocation = {}
    for agent, items in valuations.items():
        bundle = set()
        total_value = 0
        for item, val in items.items():
            if val >= threshold / 4:
                bundle.add(item)
                total_value += val
        if bundle:
            multiplier = min(round(threshold / threshold, 4), 1.0)
            allocation[agent] = f"{multiplier}*{{{', '.join(repr(item) for item in sorted(bundle))}}}"
        else:
            allocation[agent] = "0.0*{}"

    return allocation


def classify_items(valuations: Dict[str, Dict[str, float]], threshold: float) -> Tuple[Set[str], Set[str]]:
    """
    מסווגת את הפריטים ל־fat (שמנים) אם ערכם לשחקן בודד ≥ t/4, או thin (רזים) אם הערך מתחת ל - t/4.

    הסבר:
    זהו שלב 3 באלגוריתם – סיווג הפריטים לאחר נרמול.
    ננרמל את הסף כך ש־t=1.
    כל פריט שערכו לשחקן בודד הוא לפחות 1/4, נחשב ל־fat, ואחרים ל־thin.
    המטרה היא לצמצם את המורכבות ולהגביל את גודל החבילות בהיפרגרף.
    בהמשך נבנה רק קשתות מינימליות שמקיימות את התנאי הזה.

    Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "Alice": {"c1": 0.5, "c2": 0, "c3": 0},
    ...     "Bob":   {"c1": 0, "c2": 0.1, "c3": 0.2}
    ... }
    >>> fat, thin = classify_items(valuations, 1)
    >>> fat == {'c1'} and thin == {'c2', 'c3'}
    True
    """
    fat_items, thin_items = set(), set()
    for item in next(iter(valuations.values())).keys():
        max_val = max(agent_val[item] for agent_val in valuations.values())
        if max_val >= threshold / 4:
            fat_items.add(item)
        else:
            thin_items.add(item)
    return fat_items, thin_items

def build_hypergraph(valuations: Dict[str, Dict[str, float]],
                         allocation: Dict[str, List[Set[str]]],
                         fat_items: Set[str],
                         thin_items: Set[str],
                         threshold: float) -> HNXHypergraph:
    """
    בונה היפרגרף דו־צדדי, שבו קשתות הן חבילות (fat או thin) שערכן לפחות הסף הנתון.

    הסבר:
    זהו שלב 4 – בניית היפרגרף.
    נבנה גרף היפר (hypergraph) שבו:
    - צמתים בצד אחד הם השחקנים.
    - צמתים בצד השני הם הפריטים.
    - כל חבילה fat או thin שמופיעה בהקצאה מקבלת קשת בהיפרגרף.
    בפרט:
    - עבור כל fat item נבנית קשת של {i,j}.
    - עבור כל חבילה של thin items: נרצה להתאים את המתנות כך שכל הוצאת מתנה תגרור לערך הנמוך מ1 ולכן כל מתנה הכרחית.
    מטרת ההיפרגרף היא לאפשר חיפוש של התאמה מושלמת בהמשך.

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
    >>> len(hypergraph.nodes)  # מספר הצמתים
    8
    >>> len(hypergraph.edges)  # מספר הקשתות
    4
    """
    H = HNXHypergraph()
    logger.info("Building hypergraph based on allocation")

    edges = dict()
    edge_id = 0

    for player, bundles in allocation.items():
        for bundle in bundles:
            bundle_value = sum(valuations[player].get(item, 0) for item in bundle)
            if bundle_value >= threshold:
                edges[f"e{edge_id}"] = set(bundle) | {player}
                edge_id += 1

    logger.info("Building hypergraph based on allocation")
    H = HNXHypergraph(edges)
    logger.info("Hypergraph construction completed with %d nodes and %d edges", len(H.nodes), len(H.edges))
    return H


def local_search_perfect_matching(H: HNXHypergraph, valuations: Dict[str, Dict[str, float]], players: List[str], threshold: float) -> Dict[str, Set[str]]:
    """
    מבצעת חיפוש מקומי למציאת התאמה מושלמת בהיפרגרף – כל שחקן מקבל חבילה נפרדת שערכה לפחות הסף.

    הסבר:
    זהו שלב 5 – אלגוריתם חיפוש מקומי למציאת התאמה מושלמת בהיפרגרף.
    האלגוריתם בונה התאמה מושלמת תוך שימוש בעצי החלפה ובקשתות חוסמות.
    הרעיון הוא להתחיל מהתאמה ריקה ולהרחיב אותה בהדרגה:
    - בוחרים שחקן לא מותאם.
    - בונים עץ חלופי (alternating tree).
    - מחפשים קשת שאינה חוסמת ומרחיבים את ההתאמה.
    האלגוריתם מבטיח שכל שחקן יקבל קבוצה שערכה לפחות t ושאין חפיפה בין הקבוצות.

     Example 1: 2 Players, 3 Items
    >>> valuations = {
    ...     "A": {"c1": 5, "c2": 0, "c3": 4, "c4": 0},
    ...     "B": {"c1": 5, "c2": 6, "c3": 0, "c4": 0},
    ...     "C": {"c1": 0, "c2": 6, "c3": 4, "c4": 0},
    ...     "D": {"c1": 0, "c2": 0, "c3": 4, "c4": 6}
    ... }
    >>> threshold = 5
    >>> fat_items, thin_items = classify_items(valuations, threshold)
    >>> print(fat_items == {'c1', 'c2', 'c3', 'c4'})
    True


    >>> from hypernetx import Hypergraph as HNXHypergraph
    >>> edge_dict = {
    ...     "A_c3": {"A", "c3"},
    ...     "B_c1": {"B", "c1"},
    ...     "C_c2": {"C", "c2"},
    ...     "D_c4": {"D", "c4"}
    ... }
    >>> H = HNXHypergraph(edge_dict)
    >>> players = ["A", "B", "C", "D"]
    >>> best_matching = local_search_perfect_matching(H, valuations, players, threshold)
    >>> print(best_matching)
    >>> best_matching == {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
    True

    """
    from collections import deque

    matching: Dict[str, str] = {}  # player -> edge_name
    used_items: Set[str] = set()

    def is_valid_bundle(player: str, bundle: Set[str]) -> bool:
        return sum(valuations[player].get(item, 0) for item in bundle) >= threshold

    def augment_path(player: str, edge: str, parent: Dict[str, Tuple[str, str]]):
        while player in parent:
            prev_player, prev_edge = parent[player]
            matching[player] = edge
            edge = prev_edge
            player = prev_player
        matching[player] = edge
        used_items.update(H.edges[edge].elements - set(players))

    def build_alternating_tree(start_player: str) -> bool:
        queue = deque([start_player])
        parent: Dict[str, Tuple[str, str]] = {}  # player -> (parent_player, parent_edge)
        visited_players: Set[str] = {start_player}
        visited_edges: Set[str] = set()

        while queue:
            current_player = queue.popleft()
            for edge_name in H.edges:
                if edge_name in visited_edges:
                    continue

                edge_nodes = set(H.edges[edge_name].elements)
                if not edge_nodes & {current_player}:
                    continue

                bundle = edge_nodes - {current_player}
                bundle_items = bundle - set(players)
                logger.debug(f"Checking edge {edge_name} with bundle {bundle_items} for player {current_player}")
                if not is_valid_bundle(current_player, bundle_items):
                    continue

                visited_edges.add(edge_name)

                if bundle_items.isdisjoint(used_items):
                    augment_path(current_player, edge_name, parent)
                    return True

                for p, e in matching.items():
                    if not bundle_items.isdisjoint(H.edges[e].elements - set(players)):
                        if p not in visited_players:
                            visited_players.add(p)
                            parent[p] = (current_player, edge_name)
                            queue.append(p)
        return False

    for player in players:
        if player not in matching:
            if not build_alternating_tree(player):
                return {}

    # Build final allocation
    result: Dict[str, Set[str]] = {}
    for player, edge_name in matching.items():
        items = H.edges[edge_name].elements - {player}
        result[player] = items - set(players)
    return result


if __name__ == "__main__":
    # 1. Run the doctests:
    import doctest, sys
    print("\n", doctest.testmod(), "\n")

    # 2. Run the algorithm on random instances, with logging:
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
