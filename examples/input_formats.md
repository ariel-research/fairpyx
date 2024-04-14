# Input formats


```python
import fairpyx
divide = fairpyx.divide
```



`fairpyx` allows various input formats, so that you can easily use it on your own data,
whether for applications or for research.


## Valuations

Suppose you want to divide candies among your children.
It is convenient to collect their preferences in a dict of dicts:


```python
valuations = {
    "Ami": {"green": 8, "red":7, "blue": 6, "yellow": 5},
    "Tami": {"green": 12, "red":8, "blue": 4, "yellow": 2} }
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations)
```



You can then see the resulting allocation with the agents' real names:


```python
print(allocation)
```

```
{'Ami': ['blue', 'green'], 'Tami': ['red', 'yellow']}
```



For research, passing a dict of dicts as a parameter may be too verbose.
You can call the same algorithm with only the values, or only the value matrix:


```python
print(divide(fairpyx.algorithms.round_robin, valuations={"Ami": [8,7,6,5], "Tami": [12,8,4,2]}))
print(divide(fairpyx.algorithms.round_robin, valuations=[[8,7,6,5], [12,8,4,2]]))
```

```
{'Ami': [0, 2], 'Tami': [1, 3]}
{0: [0, 2], 1: [1, 3]}
```



For experiments, you can use a numpy random matrix. The code below generates random values for 5 agents and 12 courses:


```python
import numpy as np
valuations = np.random.randint(1,100,[5,12])
print(valuations)
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations)
print(allocation)
```

```
[[11 51 26 40 92 22 25  9 26 44 52 98]
 [58 55  4 57 48  3 24 24 51 84 97  2]
 [ 6  5 31 66 74 95 11 21 81 28 98 28]
 [ 9 50  8  9 37 24 42 22 90 15 36 11]
 [43 73 76 60 30 20  4 11  1 54 68 86]]
{0: [4], 1: [0, 7, 9], 2: [5, 10], 3: [6, 8], 4: [1, 2, 3, 11]}
```



## Capacities

There are several input formats for agent capacities. You can set the same capacity to all agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [4, 11], 1: [0, 9], 2: [5, 10], 3: [6, 8], 4: [1, 2]}
```



Or different capacities to different agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=[1,2,3,2,1])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [3], 1: [0, 9], 2: [4, 5, 10], 3: [1, 8], 4: [11]}
```



There are several input formats for agent capacities. You can set the same capacity to all agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [4, 11], 1: [0, 9], 2: [5, 10], 3: [6, 8], 4: [1, 2]}
```



Similarly, you can set the same capacity to all items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [1, 4, 9, 11], 1: [0, 3, 9, 10], 2: [4, 5, 8, 10], 3: [1, 5, 6,
8], 4: [0, 2, 3, 11]}
```



Or different capacities to different items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=[1,2,1,2,1,2,1,2,1,2,1,2])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [1, 4, 11], 1: [0, 3, 7, 9], 2: [3, 5, 10], 3: [5, 6, 7, 8], 4:
[1, 2, 9, 11]}
```



## Conflicts

You can specify agent_conflicts - a set of items that cannot be allocated to this agent (e.g. due to missing preliminaries):


```python
valuations = {
    "Ami": {"green": 8, "red":7, "blue": 6, "yellow": 5},
    "Tami": {"green": 12, "red":8, "blue": 4, "yellow": 2} }
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations, agent_conflicts={"Ami": ["green", "red", "blue"], "Tami": ["red", "blue", "yellow"]}) 
print(allocation)
```

```
{'Ami': ['yellow'], 'Tami': ['green']}
```



You can also specify item_conflicts - a set of items that cannot be taken together (e.g. due to overlapping times):


```python
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations, item_conflicts={"green": ["yellow", "red", "blue"]})
print(allocation)
```

```
{'Ami': ['green'], 'Tami': ['blue', 'red', 'yellow']}
```


Note that not all algorithms can handle conflicts.


---
Markdown generated automatically from [input_formats.py](input_formats.py) using [Pweave](http://mpastell.com/pweave) 0.30.3 on 2024-04-14.
