"""
An implementation of the algorithms in:
"Maximin-Aware Allocations of Indivisible Goods" by H. Chan, J. Chen, B. Li, and X. Wu (2019)
https://arxiv.org/abs/1905.09969
Programmer: Sonya Rybakov
Since: 2024-05

Disclaimer: all algorithms are on additive valuations
 """
from fairpyx import Instance, AllocationBuilder, divide


def divide_and_choose_for_three(alloc: AllocationBuilder):
    """
    Algorithm 1: Finds an mma1 allocation for 3 agents using leximin-n-partition.
    note: Only valuations are needed

    Examples:
    step 2 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [9,10], "Bob": [7,5], "Claire":[2,8]})
    {'Alice': [0], 'Bob': [1], 'Claire': []}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,8,8,7,5,2,3]}
    >>> divide(divide_and_choose_for_three, valuations=val_7items)
    {'Alice': [1,2], 'Bob': [0,5,6], 'Claire': [3,4]}

    step 3 allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [0,3], 'Bob': [1], 'Claire': [2]}

    step 4-I allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,7,6,2], "Bob": [5,5,3,7], "Claire":[2,2,2,2]})
    {'Alice': [2], 'Bob': [3], 'Claire': [0,1]}

    step 4-II allocation ok:
    >>> divide(divide_and_choose_for_three, valuations={"Alice": [2,7,6,4], "Bob": [5,5,3,7], "Claire":[2,2,2,2]})
    {'Alice': [0,1], 'Bob': [3], 'Claire': [2]}
    """
    pass
    # No need to return a value. The `divide` function returns the output.


def envy_reduction_procedure(alloc: AllocationBuilder) -> AllocationBuilder:
    """
    Procedure P for algo. 2: builds an envy graph from a given allocation, finds and reduces envy cycles.
    i.e. allocations with envy-cycles should and would be fixed here.

    :param alloc: the current allocation
    :return: updated allocation with no envy cycles

    Note: the example wouldn't provide envy cycle neccessarly,
    but it is easier to create envy than find an example with such.

    Example:
    >>> instance = Instance(valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    >>> allocator = AllocationBuilder(instance)
    >>> allocator.give('Alice', 2) # Alice envies both Claire and Bob
    >>> allocator.give('Bob', 1) # Bob envies both Alice and Claire
    >>> allocator.give('Claire', 0) # Claire envies both Alice and Bob
    >>> reduced = envy_reduction_procedure(allocator)
    >>> reduced.sorted
    {'Alice': [0], 'Bob': [2], 'Claire': [1]}
    """
    pass


def alloc_by_matching(alloc: AllocationBuilder):
    """
    Algorithm 2: Finds an 1/2mma or mmax allocation for any amount of agents and items,
    using graphs and weighted natchings.
    note: Only valuations are needed

    Examples:
    >>> divide(alloc_by_matching, valuations={"Alice": [10,10,6,4], "Bob": [7,5,6,6], "Claire":[2,8,8,7]})
    {'Alice': [0], 'Bob': [1], 'Claire': [2,3]}
    >>> val_7items = {"Alice": [10,10,6,4,2,2,2], "Bob": [7,5,6,6,6,2,9], "Claire":[2,8,8,7,5,2,3]}
    >>> divide(alloc_by_matching, valuations=val_7items)
    {'Alice': [0,1], 'Bob': [4,5,6], 'Claire': [2,3]}
    """
    pass
    # No need to return a value. The `divide` function returns the output.
