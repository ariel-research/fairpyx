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
[[33 29 31 68 28 55 87 85 51  9 62 29]
 [47 57 24 82  9 53 49  4 35 94 25 39]
 [ 9 98 13 23 21 52  1 14 33 50 60 70]
 [29 52 68 85 20 98 85 75 93 38 36 19]
 [72 23 23 34 36 71 29 13 38  2 69 91]]
{0: [6, 7], 1: [3, 9], 2: [1, 10], 3: [2, 8], 4: [0, 4, 5, 11]}
```



## Capacities

There are several input formats for agent capacities. You can set the same capacity to all agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
No leafs - removing edge (2, 11) with minimum weight 0.17
{0: [6, 7], 1: [3, 9], 2: [1, 10], 3: [5, 8], 4: [0, 11]}
```



Or different capacities to different agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=[1,2,3,2,1])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
No leafs - removing edge (0, 6) with minimum weight 0.15
{0: [7], 1: [3, 9], 2: [1, 5, 10], 3: [6, 8], 4: [11]}
```



There are several input formats for agent capacities. You can set the same capacity to all agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
No leafs - removing edge (2, 11) with minimum weight 0.17
{0: [6, 7], 1: [3, 9], 2: [1, 10], 3: [5, 8], 4: [0, 11]}
```



Similarly, you can set the same capacity to all items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
No leafs - removing edge (0, 10) with minimum weight 0.05
{0: [3, 6, 7, 8], 1: [0, 1, 3, 9], 2: [1, 9, 10, 11], 3: [5, 6, 7, 8],
4: [0, 5, 10, 11]}
```



Or different capacities to different items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=[1,2,1,2,1,2,1,2,1,2,1,2])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [3, 6, 7], 1: [1, 3, 9], 2: [1, 9, 10, 11], 3: [2, 5, 7, 8], 4:
[0, 4, 5, 11]}
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
