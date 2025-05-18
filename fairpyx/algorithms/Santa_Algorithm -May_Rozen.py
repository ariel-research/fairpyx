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
import logging

# הגדרת הלוגר
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def santa_claus_main(allocation_builder: AllocationBuilder) -> Dict[str, Set[str]]:
    """
    >>> from fairpyx import AllocationBuilder, instances
    >>> instance = Instance()
    >>> allocation_builder = AllocationBuilder(instance=instance)
    >>> # Test 1: Simple case with 2 players and 3 items
    >>> # Define valuations for players and items
    >>> allocation_builder.add_valuation("Alice", {"c1": 5, "c2": 0, "c3": 6})
    >>> allocation_builder.add_valuation("Bob", {"c1": 0, "c2": 8, "c3": 0})
    >>> result = santa_claus_main(allocation_builder)
    >>> # Expecting a matching that allocates items between Alice and Bob
    >>> result
    {'Alice': {'c1', 'c3'}, 'Bob': {'c2'}}

    >>> # Test 2: More complex case with 4 players and 4 items
    >>> allocation_builder = AllocationBuilder()
    >>> # Define valuations for 4 players and 4 items
    >>> allocation_builder.add_valuation("A", {"c1": 10, "c2": 0, "c3": 0, "c4": 6})
    >>> allocation_builder.add_valuation("B", {"c1": 10, "c2": 8, "c3": 0, "c4": 0})
    >>> allocation_builder.add_valuation("C", {"c1": 0, "c2": 8, "c3": 6, "c4": 0})
    >>> allocation_builder.add_valuation("D", {"c1": 0, "c2": 0, "c3": 6, "c4": 6})
    >>> result = santa_claus_main(allocation_builder)
    >>> # Expecting a different matching due to more players and items
    >>> result
    {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
    """
    # יצירת ה-instance כאן
    your_instance = Instance()
    # יצירת AllocationBuilder עם ה-instance
    allocation_builder = AllocationBuilder(instance=your_instance)

    # כאן אנחנו מניחים ש- allocation_builder יספק את הערכים הנדרשים כמו valuations.
    valuations = allocation_builder.to_valuations_array()
    agent_names = allocation_builder.agent_names()     # ['Alice', 'Bob', ...]
    item_names = allocation_builder.item_names()       # ['c1', 'c2', ...]

    # מיפוי בין אינדקסים לשמות
    agent_index_to_name = {i: name for i, name in enumerate(agent_names)}
    item_index_to_name = {j: name for j, name in enumerate(item_names)}

    # חיפוש בינארי על הסף
    low, high = 0, valuations.max()
    best_matching: Dict[int, Set[int]] = {}

    while high - low > 1e-4:
        mid = (low + high) / 2
        if is_threshold_feasible(valuations, mid):
            low = mid
            allocation = solve_configuration_lp(valuations, mid)
            fat_items, thin_items = classify_items(valuations, mid)
            H = build_hypergraph(valuations, allocation, fat_items, thin_items, mid)
            best_matching = local_search_perfect_matching(H)
        else:
            high = mid

    # המרה חזרה לשמות
    result: Dict[str, Set[str]] = {
        agent_index_to_name[i]: {item_index_to_name[int(s.replace("item_", ""))] for s in items}
        for i, items in best_matching.items()
    }

    return result

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


def solve_configuration_lp(valuations: Dict[str, Dict[str, float]], threshold: float) -> Dict[str, List[Set[str]]]:
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
    { 'Alice': 1*{'c1', 'c3'}, 'Bob': 1*{'c2'}

    """

    logger.debug("Applying fractional allocation with multiplier: %f", multiplier)

    fractional_allocation = {}

    for player, items in allocation.items():
        fractional_allocation[player] = {}
        for item, value in items.items():
            fractional_value = value * multiplier
            fractional_allocation[player][item] = fractional_value
            logger.debug("Player %s: Item %s value updated to %f", player, item, fractional_value)

    logger.info("Fractional allocation completed: %s", fractional_allocation)
    return fractional_allocation


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
    >>> classify_items(valuations, 1)
    ({'c1'}, {'c3', 'c2'})
    """
    logger.debug("Starting item classification with threshold %f", threshold)

    fat_items = set()
    thin_items = set()

    for player, items in valuations.items():
        for item, value in items.items():
            if value >= threshold / 4:
                fat_items.add(item)
                logger.debug("Item %s classified as fat", item)
            else:
                thin_items.add(item)
                logger.debug("Item %s classified as thin", item)

    logger.info("Classification completed. Fat items: %s, Thin items: %s", fat_items, thin_items)
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
    >>> build_hypergraph(valuations, allocation, fat_items, thin_items, 4)
    >>> len(hypergraph.nodes)  # מספר הצמתים
    8
    >>> len(hypergraph.edges)  # מספר הקשתות
    4
    """

    logger.info("Building hypergraph based on allocation")

    H = HNXHypergraph()
    for player, items in allocation.items():
        for item in items:
            if item in fat_items:
                logger.debug("Adding edge for fat item %s and player %s", item, player)
                H.add_edge([player, item])
            elif item in thin_items:
                logger.debug("Adding edge for thin item %s and player %s", item, player)
                H.add_edge([player, item])

    logger.info("Hypergraph construction completed with %d nodes and %d edges", len(H.nodes), len(H.edges))
    return H

def local_search_perfect_matching(hypergraph: HNXHypergraph) -> Dict[str, Set[str]]:
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
    >>> fat_items
    {'c1', 'c2', 'c3', 'c4'}
    >>> thin_items
    set()

    >>> allocation = {
    ...     "A": [{"c3"}],
    ...     "B": [{"c1"}],
    ...     "C": [{"c2"}],
    ...     "D": [{"c4"}]
    ... }

    >>> from hypernetx import Hypergraph as HNXHypergraph
    >>> H = build_hypergraph(valuations, allocation, fat_items, thin_items, threshold)
    >>> matching = local_search_perfect_matching(H)
    >>> matching == {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
    True
    """

    logger.info("Starting local search for perfect matching in the hypergraph")

    matching = {}

    while len(matching) < len(hypergraph.nodes):
        unmatched_players = set(hypergraph.nodes) - set(matching.keys())
        if not unmatched_players:
            logger.info("All players are matched")
            break

        player = unmatched_players.pop()
        logger.debug("Trying to find match for player %s", player)

        found_match = False
        for edge in hypergraph.edges:
            if player in edge:
                edge_players = [n for n in edge if n != player]
                if len(edge_players) == 1:
                    matching[player] = edge_players[0]
                    logger.debug("Player %s matched with item %s", player, edge_players[0])
                    found_match = True
                    break

        if not found_match:
            logger.warning("No match found for player %s", player)

    logger.info("Local search completed. Matching: %s", matching)
    return matching

if __name__ == "__main__":
    # 1. Run the doctests:
    import doctest, sys
    print("\n", doctest.testmod(), "\n")

    # 2. run with logging or real examples
    # Currently not implemented for this algorithm.
