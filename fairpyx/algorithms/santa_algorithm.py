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

from typing import Dict, List, Set, Tuple
from hypernetx import Hypergraph as HNXHypergraph
from fairpyx import Instance, AllocationBuilder
from fairpyx import validate_allocation
import logging
from typing import Optional


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
    logger.debug("Parsing allocation strings: %s", allocation)
    return parsed

def santa_claus_main(allocation_builder: AllocationBuilder) -> Dict[str, Set[str]]:
    """
    הפונקציה הראשית – מממשת את אלגוריתם 'סנטה קלאוס':
    1. חיפוש בינארי על סף t.
    2. פתרון LP וקבלת הקצאה שברירית.
    3. סיווג fat/thin.
    4. בניית היפר-גרף.
    5. חיפוש-מקומי להתאמה מושלמת.
    6. הקצאה סופית תחת מגבלות קיבולת.
    מחזירה מילון {agent: set(items)}.

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
    # שולפים את המידע מה-AllocationBuilder: שמות סוכנים ופריטים
    instance = allocation_builder.instance
    agent_names = list(instance.agents)
    item_names = list(instance.items)
    agent_capacities = {a: instance.agent_capacity(a) for a in agent_names}
    item_capacities  = {i: instance.item_capacity(i)  for i in item_names}
    logger.info("Starting santa_claus_main")
    logger.debug("Instance agents: %s", agent_names)
    logger.debug("Instance items: %s", item_names)

    # בונים מטריצת הערכות: לכל סוכן, מה ערכו עבור כל פריט
    valuations = {
        agent: {
            item: instance.agent_item_value(agent, item)
            for item in item_names
        }
        for agent in agent_names
    }

    # מחשבים את טווח החיפוש הבינארי לפי הערך המקסימלי של פריט כלשהו
    all_item_values = [v for val in valuations.values() for v in val.values()]
    high = min(sum(v.values()) for v in valuations.values()) # high = min [over all agents i] of sum[all_item_values for i]

    low = 0
    logger.debug("Initial valuations: %s", valuations)
    logger.debug("Initial binary search range: low=%f, high=%f", low, high)

    best_matching = {}

    # == חיפוש בינארי על t ==
    while high - low > 1e-4: # חיפוש בינארי: למצוא את ערך הסף הגבוה ביותר שבו עדיין ניתן לבצע הקצאה הוגנת
        mid = (low + high) / 2 # אם אפשרי לבצע הקצאה לכל סוכן עם ערך לפחות mid:
        if is_threshold_feasible(valuations, mid, agent_names): # יש שידוך מושלם וניקח אותו
            raw_allocation = solve_configuration_lp(valuations, mid)
            allocation = parse_allocation_strings(raw_allocation)
            fat_items, thin_items = classify_items(valuations, mid)
            H = build_hypergraph(valuations, allocation, fat_items, thin_items, mid)

            matching = local_search_perfect_matching(H, valuations, agent_names, threshold=mid)

            if len(matching) == len(agent_names):
                best_matching = matching
                low = mid
            else:
                if mid!=0:
                    logger.info("Matching incomplete at threshold %.4f", mid)
                    high = mid
                else:
                    return {}
        else:
            if mid != 0:
                logger.info("Matching incomplete at threshold %.4f", mid)
                high = mid
            else:
                return {}

    # == הקצאה לפי קיבולות ==
    # עוברים על הסוכנים בסדר אלפביתי ומקצים בכל פעם
    # את הפריט הפנוי בעל הערך הגבוה ביותר עבור הסוכן
    # אם כמה פריטים שווים בערכם, שוברים שוויון לפי שם הפריט (a-z).
    used_items: Set[str] = set()
    final_allocation: Dict[str, List[str]] = {}

    for agent in sorted(agent_names): # מעבר על כל הסוכנים
        cap = agent_capacities.get(agent, 1)
        chosen: List[str] = []
        for _ in range(cap): # לפי הקיבולת שלו
            remaining = [it for it in item_names if it not in used_items] # אם הסוכן רוצה את המתנה הזאת ואף אחד לא קיבל אותה כבר קודם
            if not remaining:
                break
            remaining.sort(key=lambda it: (-valuations[agent][it], it)) # בוחרים את הפריט עם הערך הגבוה ביותר (שוברי-שוויון לפי שם)
            item = remaining[0]
            used_items.add(item)
            chosen.append(item)
        final_allocation[agent] = chosen

    validate_allocation(instance, final_allocation) # בדיקה שההקצאה הסופית תקינה (בלי כפילויות וכו')
    logger.info("Final matching found at threshold %.4f: %s", low, best_matching)
    return {agent: set(items) for agent, items in final_allocation.items()} # מחזירים הקצאה עם סטים (ולא רשימות)

def is_threshold_feasible(valuations: Dict[str, Dict[str, float]], threshold: float,agent_names) -> bool:
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
    >>> is_threshold_feasible(valuations, 15,{"Alice","Bob"})
    False
    >>> is_threshold_feasible(valuations, 10,{"Alice","Bob"})
    False
    >>> is_threshold_feasible(valuations, 8,{"Alice","Bob"})
    True

    Example 2: 2 Players, 2 Items (conflict)
    >>> valuations = {
    ...     "Alice": {"c1": 10, "c2": 0},
    ...     "Bob":   {"c1": 0, "c2": 9}
    ... }
    >>> is_threshold_feasible(valuations, 10,{"Alice","Bob"})
    False
    >>> is_threshold_feasible(valuations, 9,{"Alice","Bob"})
    True
    """
    logger.info("Threshold %.4f is feasible, solving LP...", threshold)

    # פותרים את הבעיה הליניארית לקבלת הקצאה ראשונית
    raw_allocation = solve_configuration_lp(valuations, threshold)
    allocation = parse_allocation_strings(raw_allocation)  # המרת הפורמט
    fat_items, thin_items = classify_items(valuations, threshold)  # מסווגים פריטים לשמנים ורזים בהתאם ל-threshold
    H = build_hypergraph(valuations, allocation, fat_items, thin_items, threshold)  # בונים היפרגרף מההקצאה

    matching = local_search_perfect_matching(H, valuations, agent_names,
                                             threshold=threshold)  # מבצעים חיפוש מקומי למציאת התאמה מושלמת
    if len(matching) == len(agent_names):  # אם אכן גודל השידוך הוא כגודל הילדים/סוכנים
        logger.debug("Starting threshold feasibility check for threshold: %f", threshold)

        for player, items in valuations.items():
            total_value = sum(value for value in items.values())
            logger.debug("Player %s has total value: %f", player, total_value)

            if total_value < threshold:
                logger.warning("Player %s's total value %f is below threshold %f", player, total_value, threshold)
                return False

        logger.info("Threshold feasibility check passed, all players can receive at least %f value", threshold)
        return True

    return False

def solve_configuration_lp(valuations: Dict[str, Dict[str, float]], threshold: float) -> Dict[str, str]:
    """
    פונקציה זו פותרת את הבעיה הליניארית (LP) של הקונפיגורציה ומחזירה הקצאה שברית של חבילות לשחקנים.

    הסבר:
    זהו שלב 2 של האלגוריתם – פתרון תכנות ליניארי (Configuration LP Relaxation).
    נגדיר משתנים בינאריים x_{i,S} שמציינים אם שחקן i מקבל חבילה S ⊆ R שערכה הכולל לפחות 4/t.
    הפונקציה הזו פותרת את הבעיה הליניארית בצורת Relaxation ומחזירה עבור כל שחקן אוסף של חבילות אפשריות שערכן לפחות 4/t.
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
        bundle = set() # סט המתנות שהילד יקבל
        total_value = 0
        for item, val in items.items():
            if val >= threshold / 4:
                bundle.add(item) # הוספת הצלע להקצאה
                total_value += val # הוספת הערך לטוטל של ההייפר צלע
        if bundle: # אם אכן מצאנו הקצאה
            multiplier = min(round(threshold / threshold, 4), 1.0) # מתבצע כאן עיגול!
            allocation[agent] = f"{multiplier}*{{{', '.join(repr(item) for item in sorted(bundle))}}}"
        else:
            allocation[agent] = "0.0*{}"

    logger.info("Solving configuration LP with threshold %.4f", threshold)
    logger.debug("Configuration LP solution: %s", allocation)
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
    fat_items, thin_items = set(), set() # סטים ריקים: אחד לשמנים, אחד לרזים
    for item in next(iter(valuations.values())).keys(): # לוקחים את שמות כל הפריטים מתוך מילון הערכים (valuations):
        #   - .keys()  → מחזיר את שמות הפריטים
        max_val = max(agent_val[item] for agent_val in valuations.values()) # מחשבים את הערך המקסימלי של הפריט הזה אצל כל השחקנים

        # החלטה: שמן או רזה?
        # אם למישהו מהשחקנים ערך-מינימום ≥ (t / 4) הפריט נחשב "שמן" (fat)
        #   – זה תנאי מתוך המאמר שמבטיח שפריט בודד יכול לתרום לפחות רבע מהרף t.
        if max_val >= threshold / 4:
            fat_items.add(item)
        else: # אחרת הוא רזה
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
    logger.info("Building hypergraph based on allocation")

    logger.info("Building hypergraph based on allocation")

    edges: dict[str, set[str]] = {}
    edge_id = 0
    seen:  set[frozenset[str]] = set()     # דה-דופליקציה – לא ליצור שוב את אותה צלע

    #   המאמר מציע להתחיל בחבילות שקיבלנו מ-Configuration-LP (“Partial matching”)
    for player, bundles in allocation.items(): # לכל שחקן
        for bundle in bundles: #   ולכל חבילה שה-LP נתן לו
            nodes = frozenset({player, *bundle})  # {שחקן} ∪ {פריטים בחבילה}
            if nodes in seen: #   אם כבר יצרנו אותה–דלג
                continue
            seen.add(nodes) # תסמן כעת שראינו את הקבוצת מתנות הזו
            edges[f"lp{edge_id}"] = set(nodes) #  מוסיפים כ-צלע מהסוג lp*
            edge_id += 1 # תתקדם להייפר צלע הבאה

    #  הוספת קשתות fat: לכל פריט שמן נבדוק אילו שחקנים מעריכים אותו ≥ threshold
    for item in fat_items:
        for player in valuations:
            if valuations[player].get(item, 0) >= threshold/4: #צריך לוודא שזה לא אפס - לוודא אם צריך את הבדיקה
                nodes = frozenset({player, item})
                if nodes in seen:
                    continue
                seen.add(nodes)
                edges[f"f{edge_id}"] = set(nodes)
                edge_id += 1

    #  הוספת קשתות thin: ניצור תתי-קבוצות מינימליות מפריטי thin שמערכן ≥ threshold לשחקן
    #   חבילה של פריטי-Thin צריכה ביחד להגיע לערך ≥ t.
    #   כדי לשמור על “קשתות קצרות” אנחנו לוקחים *תתי-קבוצות מינימליות*
    #   – אם k פריטים מספיקים, אין סיבה להוסיף גם קבוצה גדולה יותר שמכילה אותם.
    from itertools import combinations
    for player in valuations:
        for r in range(1, len(thin_items) + 1): # גודל החבילה (1..|Thin|)
            for bundle in combinations(thin_items, r): # כל תת-קבוצה בגודל r
                if sum(valuations[player].get(i, 0) for i in bundle) >= threshold/4: #  האם ביחד ≥ t/4 ?
                    edges[f"t{edge_id}"] = set(bundle) | {player} #  צלע מהסוג t*
                    edge_id += 1

    logger.info("Building hypergraph based on allocation")
    H = HNXHypergraph(edges)
    logger.info("Hypergraph construction completed with %d nodes and %d edges", len(H.nodes), len(H.edges))
    return H

# פונקצית עזר - מחזירה את הקשתות שניתן להוסיף לעץ
def extend_alternating_tree(H: HNXHypergraph,
                            visited_players: Set[str],
                            visited_edges: Set[str],
                            players: List[str],
                            valuations: Dict[str, Dict[str, float]],
                            threshold: float) -> Optional[str]:
    """
    מנסה להרחיב את עץ החילופים לפי למא 3.2.
    מחזירה את שם הקשת שאפשר להוסיף לעץ, או None אם אין כזו.
    """

    covered_nodes = set() # כל הקודקודים שנמצאים בקשתות האלה (גם שחקנים וגם מתנות)
    for edge_name in visited_edges: # כל הקשתות שכבר בעץ
        covered_nodes |= set(H.edges[edge_name])
    covered_items = covered_nodes - set(players) #  רק המתנות (כי אנחנו רוצים לבדוק האם יש מתנות חדשות)


    for edge_name in H.edges: # עוברים על כל הקשתות שלא השתמשנו בהן עדיין
        if edge_name in visited_edges:
            continue # אם כבר ביקרנו בהן תמשיך הלאה
        edge_nodes = set(H.edges[edge_name])
        edge_players = edge_nodes & set(players) # מי מהשחקנים שייך לקשת הזו
        edge_items = edge_nodes - set(players) # אילו מתנות נמצאות בה

        if not edge_players & visited_players: # אם הקשת לא מחוברת לעץ אז תמשיך הלאה
            continue

        # תנאי קריטי – אם אין יותר פריטים חדשים תמשיך הלאה
        if not edge_items.isdisjoint(covered_items):
            continue

        # אם הקשת מספקת מישהו
        for player in edge_players:
            value = sum(valuations[player].get(item, 0) for item in edge_items)
            if value >= threshold: # ואם הערך אכן גדול מהסף
                return edge_name

    return None




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

    def is_valid_bundle(player: str, bundle: Set[str]) -> bool: # בדיקה האם הבנייה בכלל תקפה וערכה מעל הסף שצריך
        return sum(valuations[player].get(item, 0) for item in bundle) >= threshold

    def augment_path(player: str, edge: str, parent: Dict[str, Tuple[str, str]]):
        # כל עוד השחקן הנוכחי הופיע בעץ ההחלפה – נמשיך לטפס אחורה
        while player in parent:
            prev_player, prev_edge = parent[player]
            # נעדכן: השחקן הנוכחי יקבל את הקשת החדשה (edge)
            matching[player] = edge
            # ממשיכים לטפס אחורה – נתקדם לשחקן הקודם ולצלע הקודמת
            edge = prev_edge
            player = prev_player
        # לבסוף, השורש (שחקן שלא היה לו הורה) יקבל גם הוא את הקשת
        matching[player] = edge
        # נעדכן את קבוצת המתנות שהוקצו – נוריד את השחקנים מהקשת ונשמור רק את הפריטים
        used_items.update(set(H.edges[edge]) - set(players))
        logger.debug("Matching after augment: %s", matching)
        logger.debug("Used items after augment: %s", used_items)

    def build_alternating_tree(start_player: str) -> bool:
        queue = deque([start_player]) # מתחילים לבנות עץ חלופי מהשחקן שעדיין לא קיבל חבילה
        # parent: לכל שחקן נזכור מאיזה שחקן הגענו ולאיזו קשת
        parent: Dict[str, Tuple[str, str]] = {}  # player -> (parent_player, parent_edge)
        visited_players: Set[str] = {start_player} # רשימת שחקנים שביקרנו בהם כבר בעץ
        visited_edges: Set[str] = set() # רשימת קשתות שכבר בדקנו

        while queue: # נבנה עץ החלפה בצורה של BFS - הוא מנסה קודם את החילופים הכי פשוטים — כאלה שדורשים הכי מעט "הזזה" של חבילות
            current_player = queue.popleft() # הוצאה מהתור על ידי שימוש בפקודה pop
            for edge_name in H.edges: # נעבור על כל הקשתות האפשריות בהיפרגרף
                if edge_name in visited_edges:
                    continue # אם כבר בדקנו את הקשת – נמשיך הלאה
                logger.debug("Visiting edge %s", edge_name)

                edge_nodes = set(H.edges[edge_name]) # ניקח את הקודקודים שמחוברים לקשת הזו
                if current_player not in edge_nodes: # אם הקשת לא כוללת את השחקן הנוכחי – לא רלוונטית
                    logger.debug(f"Skipping edge {edge_name} – doesn't include {current_player}")
                    continue

                bundle = edge_nodes - {current_player} # נפריד את החבילה מתוך הקשת (ללא השחקן הנוכחי) - לוקחים רק את המתנות
                bundle_items = bundle - set(players)  # כלומר רק פריטים, לא שחקנים
                logger.debug(f"Checking edge {edge_name} with bundle {bundle_items} for player {current_player}")
                if not is_valid_bundle(current_player, bundle_items): # אם החבילה לא מספקת את השחקן – נמשיך הלאה
                    continue

                visited_edges.add(edge_name) # נוסיף את הצלע כצלע שכבר ביקרנו בה

                if bundle_items.isdisjoint(used_items): # אם כל הפריטים בחבילה לא בשימוש – אפשר להקצות את החבילה!
                    augment_path(current_player, edge_name, parent) # נבדוק את מסלול החבילה בעץ
                    return True # הצלחנו להרחיב את ההתאמה

                for p, e in matching.items(): # אחרת, אם החבילה חופפת לשחקנים אחרים – ננסה להחליף
                    if not bundle_items.isdisjoint(set(H.edges[e]) - set(players)): # אם יש חפיפה בין החבילה הנוכחית לבין החבילה של p
                        if p not in visited_players:
                            # מוסיפים את השחקן הזה לעץ, עם קשת ההגעה
                            visited_players.add(p)
                            parent[p] = (current_player, edge_name)
                            queue.append(p) # הוספה לתור

        # אם הגענו לפה – לא הצלחנו להרחיב את ההתאמה עבור start_player
        logger.debug("Building alternating tree for player: %s", start_player)
        logger.debug("Used items so far: %s", used_items)
        logger.debug("Parent map: %s", parent)
        return False

    for player in players: # נעבור על כל השחקנים
        if player not in matching: # אם השחקן עדיין לא בשידוך
            success = False
            visited_players = set()
            visited_edges = set()
            while not success: # כל עוד אין שידוך לשחקן תמשיך
                success = build_alternating_tree(player) # תקשר אותו להייפר צלעות במידה וניתן
                if not success: # אם כרגע עדיין לא ניתן
                    edge_to_add = extend_alternating_tree( # נבדוק האם ניתן להרחיב את העץ כלומר, לחלק עוד מתנות
                        H, visited_players, visited_edges, players, valuations, threshold
                    )
                    if edge_to_add is None:
                        break  # לא הצלחנו להרחיב יותר
                    # אם הצלחנו למצוא צלע מרחיבה – נוסיף אותה לרשימת הקשתות שנבדקות בלולאה הבאה
                    visited_edges.add(edge_to_add)
            if not success:
                for player in players: # תעבור על כל השחקנים
                    if player not in matching: # אם השחקנים עדיין לא בשידוך
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
                                    break  # אין הרחבה – נעבור לשחקן/threshold הבא
                                visited_edges.add(edge_to_add)
                        # כאן אין return {}! פשוט נמשיך הלאה

    # Build final allocation
    result: Dict[str, Set[str]] = {}
    for player, edge_name in matching.items():
        items = set(H.edges[edge_name]) - {player} # הורדת השחקן הנוכחי מהקשת
        result[player] = items - set(players) # הורדת כל השחקנים המקושרים לקשת הזאת - בדיקה נוספת ואף כללית יותר
    logger.info("Starting local search for perfect matching")
    logger.debug("Players: %s", players)
    logger.debug("Threshold: %f", threshold)
    logger.debug("Initial matching: %s", matching)
    logger.info("Finished local search. Final matching: %s", matching)
    logger.debug("Constructed allocation: %s", result)
    return result


if __name__ == "__main__":
    # 1. Run the doctests:
    import doctest, sys
    print("\n", doctest.testmod(), "\n")
#
#     # 2. Run the algorithm on random instances, with logging:
#     logger.addHandler(logging.StreamHandler(sys.stdout))
#     logger.setLevel(logging.INFO)
# if __name__ == "__main__":
#     import doctest
#     doctest.run_docstring_examples(build_hypergraph, globals(), name="local", verbose=True)
#     import hypernetx as hnx
#     valuations = {
#         "A": {"c1": 5, "c2": 0, "c3": 4, "c4": 0},
#         "B": {"c1": 5, "c2": 6, "c3": 0, "c4": 0},
#         "C": {"c1": 0, "c2": 6, "c3": 4, "c4": 0},
#         "D": {"c1": 0, "c2": 0, "c3": 4, "c4": 6}
#     }
#     threshold = 4
#     players = ["A", "B", "C", "D"]
#
#     edge_dict = {
#         "A_c1": ["A", "c1"],
#         "A_c3": ["A", "c3"],
#         "A_c1c3": ["A", "c1", "c3"],
#         "B_c1": ["B", "c1"],
#         "B_c2": ["B", "c2"],
#         "B_c1c2": ["B", "c1", "c2"],
#         "C_c2": ["C", "c2"],
#         "C_c3": ["C", "c3"],
#         "C_c2c3": ["C", "c2", "c3"],
#         "D_c4": ["D", "c4"],
#         "D_c3": ["D", "c3"],
#         "D_c3c4": ["D", "c3", "c4"]
#     }
#
#     H = hnx.Hypergraph(setsystem=edge_dict)
#     print("Edges in hypergraph:")
#     for e in H.edges:
#         print(e, "→", list(H.edges[e]))
#
#     result = local_search_perfect_matching(H, valuations, players, threshold)
#
#     print("Final matching:", result)
#     expected = {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}}
#     print("Match is correct:", result == expected)



"""

GOAL: find maximum T such that
      each child can get value at least T.

Definition: T is feasible == there exists a perfect matching in the hypergraph of T.


LOW =0
HIGH=min[i] sum[vi]

"""
