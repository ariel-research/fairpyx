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
[[10 53 17  2 33 69 83 58  6 21 64 74]
 [84 17 40  3 74 54  8 36 80 27 43 62]
 [71 93 44  8 34 37  5 11 63 91 42 18]
 [86  3 75 22 23 35 52  6 12 22 84 32]
 [ 4 63 84 36 28 32 33  5 12 49 28 53]]
{0: [6, 7], 1: [0, 4, 5, 8], 2: [1, 9], 3: [10], 4: [2, 3, 11]}
```



## Capacities

There are several input formats for agent capacities. You can set the same capacity to all agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  
print(allocation)
```

```
{0: [5, 6], 1: [4, 8], 2: [1, 9], 3: [0, 10], 4: [2, 11]}
```



Or different capacities to different agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=[1,2,3,2,1]) 
print(allocation)
```

```
{0: [6], 1: [4, 11], 2: [1, 8, 9], 3: [0, 10], 4: [2]}
```



Similarly, you can set the same capacity to all items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [5, 6, 7, 10], 1: [0, 4, 5, 8], 2: [0, 1, 8, 9], 3: [2, 6, 10,
11], 4: [1, 2, 9, 11]}
```



Or different capacities to different items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=[1,2,1,2,1,2,1,2,1,2,1,2])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [6, 7, 11], 1: [4, 5, 7, 11], 2: [1, 5, 8, 9], 3: [0, 3, 10], 4:
[1, 2, 3, 9]}
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
Markdown generated automatically from [input_formats.py](input_formats.py) using [Pweave](http://mpastell.com/pweave) 0.30.3 on 2025-01-26.
