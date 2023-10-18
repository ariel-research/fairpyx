# Fair item allocation algorithms


```python
import fairpyx
divide = fairpyx.divide
```



`fairpy` contains various algorithms for fair allocation of course-seats.
Before starting the algorithms, let us create some inputs for them.
There are two agents (students), "avi" and "beni", each of them requires three seats:

```python
agent_capacities = {"avi": 3, "beni": 3}
```



There are four items (courses), each of them has a different number of seats:

```python
item_capacities  = {"w": 2, "x": 1, "y": 2, "z": 1}
```



Students assign different values to different courses:

```python
valuations={"avi": {"x":5, "y":4, "z":3, "w":2}, "beni": {"x":5, "y":2, "z":4, "w":3}}
```



Construct the instance:

```python
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations)
```



Compute the allocation:

```python
allocation = divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
print(allocation)
```

```
{'avi': ['w', 'x', 'y'], 'beni': ['w', 'y', 'z']}
```



Try a different algorithm:

```python
allocation = divide(fairpyx.algorithms.almost_egalitarian_with_donation, instance=instance)
print(allocation)
```

```
{'avi': ['w', 'y'], 'beni': ['w', 'x', 'z']}
```



add item conflicts (- courses that cannot be taken simultaneously):

```python
item_conflicts={"x": ["w"], "w": ["x"]}
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations, item_conflicts=item_conflicts)
allocation = divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
print(allocation)
```

```
{'avi': ['x', 'y'], 'beni': ['w', 'y', 'z']}
```



add agent conflicts (- courses that cannot be taken by some agent, e.g. due to missing prerequisites):

```python
agent_conflicts={"avi": ["w"]}
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations, agent_conflicts=agent_conflicts)
allocation = divide(fairpyx.algorithms.iterated_maximum_matching_adjusted, instance=instance)
print(allocation)
```

```
{'avi': ['y', 'z'], 'beni': ['w', 'x', 'y']}
```


---
Markdown generated automatically from [courses.py](courses.py) using [Pweave](http://mpastell.com/pweave) 0.30.3 on 2023-10-18.
