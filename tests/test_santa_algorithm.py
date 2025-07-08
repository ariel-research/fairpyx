"""
An implementation of the algorithms in:
"Santa Claus Meets Hypergraph Matchings",
by ARASH ASADPOUR - New York University, URIEL FEIGE - The Weizmann Institute, AMIN SABERI - Stanford University,
https://dl.acm.org/doi/abs/10.1145/2229163.2229168
Programmers: May Rozen
Date: 2025-04-23
"""
import unittest
import fairpyx

class TestSantaClausAlgorithm(unittest.TestCase):
    # --------------------------Test 1: Simple case with 2 players and 3 items--------------------------
    def test_santa_claus_main_simple1(self):
        instance = fairpyx.Instance(
            valuations={"Alice": {"c1": 5, "c2": 0, "c3": 6}, "Bob": {"c1": 0, "c2": 8, "c3": 0}},
            agent_capacities={"Alice": 2, "Bob": 1},
            item_capacities={"c1": 5, "c2": 8, "c3": 6})
        allocation = fairpyx.divide(algorithm=fairpyx.algorithms.santa_claus_main, instance=instance)

        # Expecting a matching that allocates items between Alice and Bob
        assert allocation == {'Alice': ['c1', 'c3'], 'Bob': ['c2']}

    # --------------------------Test 2: More complex case with 4 players and 4 items--------------------------
    def test_santa_claus_main_simple2(self):
        instance = fairpyx.Instance(
             valuations = {"A": {"c1": 10, "c2": 0, "c3": 0, "c4": 6}, "B": {"c1": 10, "c2": 8, "c3": 0, "c4": 0},
                           "C": {"c1": 10, "c2": 8, "c3": 0, "c4": 0}, "D": {"c1": 0, "c2": 0, "c3": 6, "c4": 6}},
             agent_capacities = {"A": 1, "B": 1, "C": 1, "D": 1},
             item_capacities = {"c1": 1, "c2": 1, "c3": 1, "c4": 1})
        allocation = fairpyx.divide(algorithm=fairpyx.algorithms.santa_claus_main, instance=instance)

        # Expecting a different matching due to more players and items
        assert allocation == {'A': ['c1'], 'B': ['c2'], 'C': ['c3'], 'D': ['c4']}

    # --------------------------Test 3: Large scale case with n players and n items--------------------------
    def test_santa_claus_main_Large1(self):
        n = 10
        valuations = {
            f"Player_{i + 1}": {f"c{j + 1}": (1 if j == i else 0) for j in range(n)}
            for i in range(n)
        }

        agent_caps = {p: 1 for p in valuations}
        item_caps = {f"c{i + 1}": 1 for i in range(n)}

        instance = fairpyx.Instance(
            valuations=valuations,
            agent_capacities=agent_caps,
            item_capacities=item_caps
        )

        alloc = fairpyx.divide(
            algorithm=fairpyx.algorithms.santa_claus_main,
            instance=instance
        )

        expected = {f"Player_{i + 1}": [f"c{i + 1}"] for i in range(n)}
        assert alloc == expected

        # Verify there are no duplicate allocations (no player gets the same item as another player)
        self.assertTrue(all(len(set(items)) == len(items) for items in alloc.values()))

        # Verify the number of players matches the number of allocations
        self.assertEqual(len(alloc), n)


    # --------Test 4: case for n players and 2n presents (each player gets one present with value 40 and one with value 60)--------
    def test_santa_claus_main_Large2(self):
        # Define number of players and items
        n_players = 6
        n_items = 2 * n_players

        # Build valuations: each player values exactly two items (40 and 60)
        valuations = {
            f"Player_{i + 1}": {
                f"c{j + 1}": (40 if j == 2 * i else 60 if j == 2 * i + 1 else 0)
                for j in range(n_items)
            }
            for i in range(n_players)
        }

        # Set capacities: each player can receive 2 items, each item can be assigned once
        agent_caps = {f"Player_{i + 1}": 2 for i in range(n_players)}
        item_caps = {f"c{j + 1}": 1 for j in range(n_items)}

        # Create the instance with valuations and capacities
        instance = fairpyx.Instance(
            valuations=valuations,
            agent_capacities=agent_caps,
            item_capacities=item_caps
        )

        # Run the Santa Claus algorithm
        alloc = fairpyx.divide(
            algorithm=fairpyx.algorithms.santa_claus_main,
            instance=instance
        )

        # Build the expected allocation mapping each player to their two valued items
        expected = {
            f"Player_{i + 1}": {f"c{2 * i + 1}", f"c{2 * i + 2}"}
            for i in range(n_players)
        }
        for player, got in alloc.items():
            self.assertEqual(set(got), expected[player])

        # total number of players and no duplicate items globally
        self.assertEqual(len(alloc), n_players)

        all_items = [item for items in alloc.values() for item in items]
        self.assertEqual(len(all_items), len(set(all_items)))


if __name__ == "__main__":
    import time, sys

    start = time.perf_counter()
    # exit=False allows us to continue running code even after the tests have been executed
    result = unittest.main(exit=False)
    end = time.perf_counter()

    total = end - start
    print(f"\nTotal test suite runtime: {total:.4f} seconds")

    # If there are failures, we will return an error
    sys.exit(not result.result.wasSuccessful())
