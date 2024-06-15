"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

from fairpyx.algorithms.CM import A_CEEI
from fairpyx.algorithms.CM import remove_oversubscription
from fairpyx.algorithms.CM import reduce_undersubscription

def course_match_algorithm(alloc: AllocationBuilder, budget: dict):
    """
    Perform the Course Match algorithm to find the best course allocations.
    
    :param alloc: (AllocationBuilder) an allocation builder object

    :return: (dict) course allocations

    :example
    >>> instance = Instance(
    ...   agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ...   item_conflicts = {"c1": [], "c2": [], "c3": []},
    ...   agent_capacities = {"Alice": 1, "Bob": 1, "Tom": 1}, 
    ...   item_capacities  = {"c1": 1, "c2": 1, "c3": 1},
    ...   valuations = {"Alice": {"c1": 100, "c2": 0, "c3": 0},
    ...                 "Bob": {"c1": 0, "c2": 100, "c3": 0},
    ...                 "Tom": {"c1": 0, "c2": 0, "c3": 100}
    ... })
    >>> budget = {"Alice": 1.0, "Bob": 1.1, "Tom": 1.3}    
    >>> allocation = AllocationBuilder(instance)
    >>> course_match_algorithm(allocation, budget)
    {'Alice': ['c1'], 'Bob': ['c2'], 'Tom': ['c3']}

    >>> instance = Instance(
    ...   agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ...   item_conflicts = {"c1": [], "c2": [], "c3": []},
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2}, 
    ...   item_capacities  = {"c1": 2, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 100, "c2": 100, "c3": 0},
    ...                 "Bob": {"c1": 0, "c2": 100, "c3": 100},
    ...                 "Tom": {"c1": 100, "c2": 0, "c3": 100}
    ... })
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}    
    >>> allocation = AllocationBuilder(instance)
    >>> course_match_algorithm(allocation, budget)
    {'Alice': ['c1', 'c2'], 'Bob': ['c2', 'c3'], 'Tom': ['c1', 'c3']}

    # >>> instance = Instance(
    # ...   agent_conflicts = {"Alice": [], "Bob": []},
    # ...   item_conflicts = {"c1": [], "c2": [], "c3": []},
    # ...   agent_capacities = {"Alice": 2, "Bob": 1},
    # ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    # ...   valuations = {"Alice": {"c1": 100, "c2": 60, "c3": 0},
    # ...                 "Bob": {"c1": 0, "c2": 100, "c3": 0},
    # ... })
    # >>> allocation = AllocationBuilder(instance)
    # >>> budget = {"Alice": 3.0, "Bob": 1.0}    
    # >>> course_match_algorithm(allocation, budget)
    # {'Alice': ['c1', 'c2'], 'Bob': ['c2']}
    
    """

    price_vector = A_CEEI.A_CEEI(alloc,budget,15)
    # print("After A_CEEI", price_vector)
    price_vector = remove_oversubscription.remove_oversubscription(alloc, price_vector, budget)
    # print("After remove",price_vector)
    reduce_undersubscription.reduce_undersubscription(alloc, price_vector, budget)
    # print("After reduce, alloc.bundles-",alloc.bundles)
    #cheng alloc.bundles to de {agent:[]}
    CM = {agent:list(alloc.bundles[agent]) for agent in alloc.bundles}
    #sort the bundles
    for agent in CM:
        CM[agent].sort()
    print(CM)
   


if __name__ == "__main__":
    import doctest
    doctest.testmod()

