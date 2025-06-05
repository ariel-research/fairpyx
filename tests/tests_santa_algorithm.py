import numpy as np
from fairpyx.algorithms.Santa_Algorithm import (
    is_threshold_feasible,
    solve_configuration_lp,
    classify_items,
    build_hypergraph,
    local_search_perfect_matching,
    Hypergraph
)

import time
seed = time.time_ns()
print("seed = ",seed)
np.random.seed(seed)

# ========== Basic edge cases ==========

# Test: Empty input should not be feasible
def test_empty_input():
    valuations = np.array([[]])
    assert not is_threshold_feasible(valuations, 0.5)


# Test: Single player and single item enough to meet threshold
def test_one_player_one_item_enough():
    valuations = np.array([[1.0]])
    assert is_threshold_feasible(valuations, 0.5)


# Test: Single player and single item not enough to meet threshold
def test_one_player_one_item_not_enough():
    valuations = np.array([[0.2]])
    assert not is_threshold_feasible(valuations, 0.5)


# Test: Negative valuations, ensure function handles properly
def test_negative_values():
    valuations = np.array([[0.3, -0.2], [-0.1, 0.5]])
    assert is_threshold_feasible(valuations, 0.3)  # Player 2 saves it

# Test: player with only negative valuations cannot be satisfied
def test_player_with_only_negative_items():
    valuations = np.array([[-0.5, -0.3, -0.9]])
    assert not is_threshold_feasible(valuations, 0.1)

# Test: very small threshold close to zero should be feasible
def test_very_small_threshold():
    valuations = np.array([[0.0002, 0.0003]])
    assert is_threshold_feasible(valuations, 0.0001)

# Test: exact threshold value is considered feasible
def test_exact_threshold_value():
    valuations = np.array([[0.4, 0.1]])
    assert is_threshold_feasible(valuations, 0.5)


# ========== Larger input tests ==========

# Test: Large input random valuations
def test_large_input_feasibility():
    np.random.seed(0)
    valuations = np.random.rand(50, 100)  # 50 players, 100 items
    assert isinstance(is_threshold_feasible(valuations, 0.5), bool)


# Test: Very large input - stress test
def test_very_large_input():
    np.random.seed(42)
    valuations = np.random.rand(200, 500)  # 200 players, 500 items
    threshold = 0.4
    assert isinstance(is_threshold_feasible(valuations, threshold), bool)


# ========== Conflict and matching tests ==========

# Test: Conflict over valuable items (both want the same)
def test_conflict_case_no_feasible_allocation():
    valuations = np.array([
        [0.9, 0.2],
        [0.9, 0.2]
    ])
    assert not is_threshold_feasible(valuations, 0.9)


# Test: Conflict but lower threshold makes feasible
def test_conflict_case_with_lower_threshold():
    valuations = np.array([
        [0.9, 0.2],
        [0.9, 0.2]
    ])
    assert is_threshold_feasible(valuations, 0.5)


# ========== Functionality and consistency tests ==========

# Test: Random allocation followed by classification and matching consistency
def test_random_allocation_classification_consistency():
    np.random.seed(1)
    valuations = np.random.rand(5, 10)
    threshold = 0.6
    allocation = solve_configuration_lp(valuations, threshold)
    fat, thin = classify_items(valuations, threshold)
    H = build_hypergraph(valuations, allocation, fat, thin, threshold)
    match = local_search_perfect_matching(H)

    # Check that each player achieves at least the threshold value
    for player, items in match.items():
        total_value = sum(valuations[player - 1, np.array(list(items)) - 1])
        assert total_value >= threshold, f"Player {player} got {total_value}, expected at least {threshold}"


# Test: Manual classification of fat and thin items
def test_classify_items_fat_thin():
    valuations = np.array([
        [0.9, 0.1],
        [0.2, 0.9]
    ])
    fat, thin = classify_items(valuations, 0.9)
    assert fat == {1, 2}
    assert thin == set()


# Test: No fat items, all thin
def test_all_thin_items():
    valuations = np.array([
        [0.5, 0.4],
        [0.3, 0.5]
    ])
    fat, thin = classify_items(valuations, 0.7)
    assert fat == set()
    assert thin == {1, 2}


# ========== Specific hypergraph and matching tests ==========

# Test: Build hypergraph and validate its structure
def test_build_hypergraph_structure():
    valuations = np.array([
        [0.6, 0.4, 0.5],
        [0.7, 0.2, 0.4]
    ])
    allocation = {1: [{1, 2}], 2: [{3}]}
    fat, thin = classify_items(valuations, 0.5)
    H = build_hypergraph(valuations, allocation, fat, thin, 0.5)

    # Ensure nodes and edges exist
    assert H is not None
    assert hasattr(H, 'edges')
    assert len(H.edges) > 0


# Test: Perfect matching correctness
def test_local_search_perfect_matching_correctness():
    H = Hypergraph()
    H.add_edge(1, {1})
    H.add_edge(2, {2})
    match = local_search_perfect_matching(H)

    # Ensure that all players are matched
    assert 1 in match
    assert 2 in match

    # Ensure that matched bundles are disjoint
    all_items = set()
    for items in match.values():
        assert all_items.isdisjoint(items), "Matched bundles overlap"
        all_items.update(items)


# Test: Matching with minimal bundles
def test_local_search_minimal_bundles():
    H = Hypergraph()
    H.add_edge(1, {1, 2})
    H.add_edge(2, {3})
    match = local_search_perfect_matching(H)

    # Ensure each player matched
    assert set(match.keys()) == {1, 2}

    # Ensure disjointness of assigned bundles
    all_items = set()
    for player, items in match.items():
        assert all_items.isdisjoint(items), "Matched bundles overlap"
        all_items.update(items)

        # Ensure every player is matched to a non-empty set
        assert len(items) > 0, f"Player {player} was assigned an empty bundle"


# ========== Edge case: All valuations are zero ==========

# Test: All valuations are zero (no feasible allocation)
def test_all_zero_valuations():
    valuations = np.zeros((4, 5))
    assert not is_threshold_feasible(valuations, 0.1)


# ========== Edge case: Threshold higher than any possible valuation ==========

# Test: Threshold is too high for any bundle
def test_threshold_too_high():
    valuations = np.array([
        [0.2, 0.1],
        [0.3, 0.4]
    ])
    assert not is_threshold_feasible(valuations, 1.0)

# ========== Stress tests ==========
# Test: large matrix with big negative and positive values
def test_large_input_with_negatives():
    valuations = np.random.randn(300, 600) * 5  # Both negative and positive values
    assert isinstance(is_threshold_feasible(valuations, 2.5), bool)

# Test: very small numbers to check numerical stability
def test_very_small_numbers_matrix():
    valuations = np.full((10, 10), 1e-5)
    assert is_threshold_feasible(valuations, 5e-5)


