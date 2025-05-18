"""
An implementation of the algorithms in:
"Santa Claus Meets Hypergraph Matchings",
by ARASH ASADPOUR - New York University, URIEL FEIGE - The Weizmann Institute, AMIN SABERI - Stanford University,
https://dl.acm.org/doi/abs/10.1145/2229163.2229168
Programmers: May Rozen
Date: 2025-04-23
"""
import unittest
from fairpyx import AllocationBuilder, Instance
from Santa_Algorithm import santa_claus_main

from fairpyx.fairpyx.algorithms.Santa_Algorithm import is_threshold_feasible, classify_items, solve_configuration_lp, \
    build_hypergraph


class TestSantaClausAlgorithm(unittest.TestCase):
    def test_santa_claus_main_simple(self):
        # Test 1: Simple case with 2 players and 3 items
        instance = Instance()
        allocation_builder = AllocationBuilder(instance=instance)
        allocation_builder.add_valuation("Alice", {"c1": 5, "c2": 0, "c3": 6})
        allocation_builder.add_valuation("Bob", {"c1": 0, "c2": 8, "c3": 0})

        result = santa_claus_main(allocation_builder)

        # Expecting a matching that allocates items between Alice and Bob
        self.assertEqual(result, {'Alice': {'c1', 'c3'}, 'Bob': {'c2'}})

        # Test 2: More complex case with 4 players and 4 items
        instance = Instance()
        allocation_builder = AllocationBuilder(instance=instance)

        allocation_builder.add_valuation("A", {"c1": 10, "c2": 0, "c3": 0, "c4": 6})
        allocation_builder.add_valuation("B", {"c1": 10, "c2": 8, "c3": 0, "c4": 0})
        allocation_builder.add_valuation("C", {"c1": 0, "c2": 8, "c3": 6, "c4": 0})
        allocation_builder.add_valuation("D", {"c1": 0, "c2": 0, "c3": 6, "c4": 6})

        result = santa_claus_main(allocation_builder)

        # Expecting a different matching due to more players and items
        self.assertEqual(result, {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}})

        # Test 3: Large scale case with 100 players and 100 items
        instance = Instance()
        allocation_builder = AllocationBuilder(instance=instance)

        for i in range(100):
            valuations = {f"c{j + 1}": (1 if j == i else 0) for j in range(100)}  # Player_i values only item_i
            allocation_builder.add_valuation(f"Player_{i + 1}", valuations)

        result = santa_claus_main(allocation_builder)

        # Expecting each player to get exactly their corresponding item
        for i in range(100):
            self.assertEqual(result[f"Player_{i + 1}"], {f"c{100 - i}"})

        # Verify there are no duplicate allocations (no player gets the same item as another player)
        self.assertTrue(all(len(set(items)) == len(items) for items in result.values()))

        # Verify the number of players matches the number of allocations
        self.assertEqual(len(result), 100)

        # Test case for 100 players and 200 presents (each player gets one present with value 40 and one with value 60)
        instance = Instance()
        allocation_builder = AllocationBuilder(instance=instance)

        # Define valuations for 100 players and 200 presents (one with value 40 and one with value 60)
        for i in range(100):
            valuations = {f"c{j + 1}": (40 if j == i * 2 else 60) for j in
                          range(200)}  # Each player gets 1 present with value 40 and 1 present with value 60
            allocation_builder.add_valuation(f"Player_{i + 1}", valuations)

        result = santa_claus_main(allocation_builder)

        # Expecting each player to get exactly one present with value 40 and one with value 60
        for i in range(100):
            player_presents = result[f"Player_{i + 1}"]
            self.assertEqual(len(player_presents), 2)  # Ensure each player gets exactly 2 presents
            self.assertTrue(
                any(present.startswith('c') for present in player_presents))  # Ensure the presents have correct ids
            self.assertTrue(all(valuation in [40, 60] for present, valuation in
                                player_presents))  # Ensure the presents' values are 40 and 60

        # Verify the number of players matches the number of allocations
        self.assertEqual(len(result), 100)

        # Verify no duplicates in the allocation
        all_presents = [present for presents in result.values() for present in presents]
        self.assertEqual(len(all_presents),
                         len(set(all_presents)))  # Ensure no presents are assigned to multiple players

    def test_is_threshold_feasible(self):
        # Test the threshold feasibility function
        valuations = {
            "Alice": {"c1": 7, "c2": 0, "c3": 4},
            "Bob": {"c1": 0, "c2": 8, "c3": 0}
        }

        self.assertFalse(is_threshold_feasible(valuations, 15))
        self.assertFalse(is_threshold_feasible(valuations, 10))
        self.assertTrue(is_threshold_feasible(valuations, 8))

        valuations = {
            "A": {"c1": 10, "c2": 0, "c3": 0, "c4": 6},
            "B": {"c1": 10, "c2": 8, "c3": 0, "c4": 0},
            "C": {"c1": 0, "c2": 8, "c3": 6, "c4": 0},
            "D": {"c1": 0, "c2": 0, "c3": 6, "c4": 6}
        }
        self.assertTrue(is_threshold_feasible(valuations, 6))
        self.assertFalse(is_threshold_feasible(valuations, 7))


    def test_solve_configuration_lp(self):
        # Test: 2 Players, 2 Items (conflict)
        valuations = {
            "Alice": {"c1": 10, "c2": 5},
            "Bob": {"c1": 10, "c2": 5}
        }

        result = solve_configuration_lp(valuations, 5)

        # Expected result: both Alice and Bob get half of both items
        expected_result = {'Alice': {('c1', 0.5), ('c2', 0.5)}, 'Bob': {('c1', 0.5), ('c2', 0.5)}}

        self.assertEqual(result, expected_result)

    def test_classify_items(self):
        # Test the classify_items function
        valuations = {
            "Alice": {"c1": 0.9, "c2": 0.2},
            "Bob":   {"c1": 0.9, "c2": 0.2}
        }

        fat_items, thin_items = classify_items(valuations, 0.4)

        self.assertEqual(fat_items, {'c1'})
        self.assertEqual(thin_items, {'c2'})

    def test_build_hypergraph(self):
        # Test the build_hypergraph function
        valuations = {
            "A": {"c1": 10, "c2": 0, "c3": 0, "c4": 0},
            "B": {"c1": 0, "c2": 8, "c3": 0, "c4": 0},
            "C": {"c1": 0, "c2": 0, "c3": 6, "c4": 0},
            "D": {"c1": 0, "c2": 0, "c3": 0, "c4": 4}
        }

        allocation = {
            "A": [{"c1"}],
            "B": [{"c2"}],
            "C": [{"c3"}],
            "D": [{"c4"}]
        }

        fat_items, thin_items = classify_items(valuations, 4)

        # Test hypergraph creation
        hypergraph = build_hypergraph(valuations, allocation, fat_items, thin_items, 4)

        # Verify the number of nodes (players + items)
        self.assertEqual(len(hypergraph.nodes), 8)

        # Verify the number of edges
        self.assertEqual(len(hypergraph.edges), 4)

        # Check that the hypergraph contains the correct edges
        self.assertTrue(any("A" in edge and "c1" in edge for edge in hypergraph.edges))
        self.assertTrue(any("B" in edge and "c2" in edge for edge in hypergraph.edges))
        self.assertTrue(any("C" in edge and "c3" in edge for edge in hypergraph.edges))
        self.assertTrue(any("D" in edge and "c4" in edge for edge in hypergraph.edges))

    def test_local_search_perfect_matching(self):
        # Test the local_search_perfect_matching function
        valuations = {
            "A": {"c1": 5, "c2": 0, "c3": 4, "c4": 0},
            "B": {"c1": 5, "c2": 6, "c3": 0, "c4": 0},
            "C": {"c1": 0, "c2": 6, "c3": 4, "c4": 0},
            "D": {"c1": 0, "c2": 0, "c3": 4, "c4": 6}
        }

        threshold = 5
        fat_items, thin_items = classify_items(valuations, threshold)

        allocation = {
            "A": [{"c3"}],
            "B": [{"c1"}],
            "C": [{"c2"}],
            "D": [{"c4"}]
        }

        hypergraph = build_hypergraph(valuations, allocation, fat_items, thin_items, threshold)

        matching = local_search_perfect_matching(hypergraph)

        # Expecting a perfect matching
        self.assertEqual(matching, {'A': {'c1'}, 'B': {'c2'}, 'C': {'c3'}, 'D': {'c4'}})


if __name__ == "__main__":
    unittest.main()
