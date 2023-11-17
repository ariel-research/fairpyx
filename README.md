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

To activate a fair division algorithm, first construct a `fairpyx.instance`:

    import fairpyx
    valuations = {"Alice": {"w":11,"x":22,"y":44,"z":0}, "George": {"w":22,"x":11,"y":66,"z":33}}
    instance = fairpyx.Instance(valuations=valuations)

An instance can have other fields, such as: agent capacities, item capacities, agent conflicts and item conflicts. These fields are used by some of the algorithms. See [instances.py](fairpyx/instance.py) for details.

Then, use the function `fairpyx.divide` to run an algorithm on the instance. For example:

    allocation = fairpyx.divide(algorithm=fairpyx.algorithms.iterated_maximum_matching, instance=instance)
    print(allocation)

## Features and Examples

1. [Course allocation algorithms](examples/courses.md);

1. [Various input formats](examples/input_formats.md), to easily use by both researchers and end-users;


## Contributing new algorithms

1. Fork `fairpyx` and install your fork locally as follows:

    ```
    clone https://github.com/<your-username>/fairpyx.git
    cd fairpyx
    pip install -e .
    ```

2. Write a function that accepts a parameter of type `AllocationBuilder`, as well as any custom parameters your algorithm needs. The `AllocationBuilder` argument sent to your function is already initialized with an empty allocation. Your function has to modify this argument using the method `give`, which gives an item to an agent and updates the capacities. Your function need not return any value; the allocation is read from the modified parameter. See:

* [picking_sequence.py](fairpyx/algorithms/picking_sequence.py) and [iterated_maximum_matching.py](fairpyx/algorithms/iterated_maximum_matching.py) for examples of algorithms;
* [allocations.py](fairpyx/allocations.py) for more details on the `AllocationBuilder` object.

## See also

* [fairpy](https://github.com/erelsgl/fairpy) is an older library with the same goals. It contains more algorithms for fair item allocation, as well as algorithms for fair cake-cutting. `fairpyx` was created in order to provide a simpler interface, that also allows capacities and conflicts, which are important for fair course allocation.
* [Other open-source projects related to fairness](related.md).


