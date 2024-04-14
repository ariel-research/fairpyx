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
[[96 51 98 31  1 62 36 50 21 47 97 29]
 [58 91 31 79 42 60 49 86 64 88 39 56]
 [88 81 73 16 67 65 92 26 52  3 60  6]
 [91 55 72 43 10 61 97 21 89 97 69 35]
 [ 9 96 39 82 59 68 52 78 93 87 84 26]]
{0: [2], 1: [7, 11], 2: [0, 4], 3: [6, 9], 4: [1, 3, 5, 8, 10]}
```



## Capacities

There are several input formats for agent capacities. You can set the same capacity to all agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=2)  
print(allocation)
```

```
{0: [2, 10], 1: [5, 7], 2: [0, 6], 3: [8, 9], 4: [1, 3]}
```



Or different capacities to different agents:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=[1,2,3,2,1]) 
print(allocation)
```

```
{0: [10], 1: [1, 7], 2: [0, 2, 4], 3: [6, 9], 4: [8]}
```



Similarly, you can set the same capacity to all items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=2)  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [1, 2, 5, 10], 1: [1, 3, 7, 9], 2: [0, 2, 4, 6], 3: [0, 6, 8, 9],
4: [3, 7, 8, 10]}
```



Or different capacities to different items:


```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_allocation, valuations=valuations, agent_capacities=4, item_capacities=[1,2,1,2,1,2,1,2,1,2,1,2])  # , explanation_logger=fairpyx.ConsoleExplanationLogger()
print(allocation)
```

```
{0: [0, 2, 10], 1: [3, 7, 11], 2: [1, 4, 5, 6], 3: [5, 8, 9, 11], 4:
[1, 3, 7, 9]}
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
