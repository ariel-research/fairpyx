# fairpyx

![PyTest result](https://github.com/ariel-research/fairpyx/workflows/pytest/badge.svg)
[![PyPI version](https://badge.fury.io/py/fairpyx.svg)](https://badge.fury.io/py/fairpyx)

`fairpyx` is a Python library containing various algorithms for fair allocation, with an emphasis on [Course allocation](https://en.wikipedia.org/wiki/Course_allocation). It is designed for three target audiences:

* Laypeople, who want to use existing fair division algorithms for real-life problems.
* Researchers, who develop new fair division algorithms and want to quickly implement them and compare to existing algorithms.
* Students, who want to trace the execution of algorithms to understand how they work.

## Installation

For the stable version:

    pip install fairpyx

For the latest version:

    pip install git+https://github.com/ariel-research/fairpyx.git

To verify that everything was installed correctly, run one of the example programs, e.g.

    cd fairpyx
    python examples/courses.py
    python examples/input_formats.py

or run the tests:

    pytest

## Usage

To activate a fair division algorithm, first construct a `fairpyx.Instance`, for example:

    import fairpyx
    valuations = {"Alice": {"w":11,"x":22,"y":44,"z":0}, "George": {"w":22,"x":11,"y":66,"z":33}}
    instance = fairpyx.Instance(valuations=valuations)

An instance can have other fields, such as: `agent_capacities`, `item_capacities`, `agent_conflicts` and `item_conflicts`. These fields are used by some of the algorithms. See [instances.py](fairpyx/instances.py) for details.

Then, use the function `fairpyx.divide` to run an algorithm on the instance. For example:

    allocation = fairpyx.divide(algorithm=fairpyx.algorithms.iterated_maximum_matching, instance=instance)
    print(allocation)

## Features and Examples

1. [Course allocation algorithms](examples/courses.md);

1. [Various input formats](examples/input_formats.md), to easily use by both researchers and end-users;

1. [A demo of a the simple round-robin algorithm](examples/picking_sequence_demo.md);



## Contributing new algorithms

You are welcome to add fair allocation algorithms, including your published algorithms, to `fairpyx`. Please use the following steps to contribute:

1. Fork the repository, then install your fork locally as follows:

    ```
    clone https://github.com/<your-username>/fairpyx.git
    cd fairpyx
    pip install -e .
    ```

2. Read the code at [algorithm_examples.py](fairpyx/algorithms/algorithm_examples.py) to see how the implementation works. 

  * Note that the implementation does not use the `Instance` variable directly - it uses an `AllocationBuilder` variable, which tracks both the ongoing allocation and the remaining input (the remaining capacities of agents and items).

3. Write a function that accepts a parameter of type `AllocationBuilder`, as well as any custom parameters your algorithm needs. 

  * The `AllocationBuilder` argument sent to your function is already initialized with an empty allocation. 
  Your function has to modify this argument using the methods `give` or `give_bundle`, which give an item or a set of items to an agent and update the capacities accordingly. 
  * You can easily chain algorithms. For example, if the last phase of your algorithm is dividing the remaining items using round-robin, you can simply call `round_robin(alloc)` at the end of your function; the AllocationBundle object already tracks the remaining items for you.
  * Your function need not return any value; the allocation is read from the `alloc`.
  * The `divide` function is responsible for converting the `Instance` to an `AllocationBuilder` before your function starts, and extracting the allocation from the `AllocationBuilder` after your function ends, so you can focus on writing the algorithm itself.


See [allocations.py](fairpyx/allocations.py) for more details on the `AllocationBuilder` object.

## See also

* [fairpy](https://github.com/erelsgl/fairpy) is an older library with the same goals. It contains more algorithms for fair item allocation, as well as algorithms for fair cake-cutting. `fairpyx` was created in order to provide a simpler interface, that also allows capacities and conflicts, which are important for fair course allocation.
* [Other open-source projects related to fairness](related.md).


