"""
dict_product: a cartesian product of a dict.

Author: Erel Segal-Halevi
Since: 2022-05
"""

from typing import Dict, List, Any
import itertools

def dict_product(input:Dict[Any,List[Any]])->List[Dict[Any,Any]]:
    """
    Gets a dict mapping each key to a list of possible values.
    Generates all possible dicts, mapping each key to a single value.
    If, for each key k, there are nk possible values, then the number of dicts is prod_k(nk).
    >>> for x in dict_product({"x": [1,2,3], "y": [4,5], "z": [6]}): print(x)
    {'x': 1, 'y': 4, 'z': 6}
    {'x': 1, 'y': 5, 'z': 6}
    {'x': 2, 'y': 4, 'z': 6}
    {'x': 2, 'y': 5, 'z': 6}
    {'x': 3, 'y': 4, 'z': 6}
    {'x': 3, 'y': 5, 'z': 6}
    """
    keys  =  list(input.keys())
    value_lists = input.values()
    num_keys = len(keys)
    value_combinations = itertools.product(*value_lists)
    for combination in value_combinations:
        yield {keys[i]: combination[i] for i in range(num_keys)}


if __name__=="__main__":
    import doctest
    print(doctest.testmod())
    # for x in dict_product({"x": [1,2,3], "y": [4,5], "z": [6]}): print(x)

