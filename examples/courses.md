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
allocation = divide(fairpyx.algorithms.iterated_maximum_matching, instance=instance)
print(allocation)
```

```
{'avi': ['w', 'x', 'y'], 'beni': ['w', 'y', 'z']}
```



add item conflicts (- courses that cannot be taken simultaneously):

```python
item_conflicts={"x": ["w"], "w": ["x"]}
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations, item_conflicts=item_conflicts)
allocation = divide(fairpyx.algorithms.iterated_maximum_matching, instance=instance)
print(allocation)
```

```
{'avi': ['x', 'y'], 'beni': ['w', 'y', 'z']}
```



add agent conflicts (- courses that cannot be taken by some agent, e.g. due to missing prerequisites):

```python
agent_conflicts={"avi": ["w"]}
instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=item_capacities, valuations=valuations, agent_conflicts=agent_conflicts)
allocation = divide(fairpyx.algorithms.iterated_maximum_matching, instance=instance)
print(allocation)
```

```
{'avi': ['y', 'z'], 'beni': ['w', 'x', 'y']}
```



create a random instance:

```python
random_instance = fairpyx.Instance.random_uniform(
    num_of_agents=10, num_of_items=3, agent_capacity_bounds=[2,3], item_capacity_bounds=[5,7], 
    item_base_value_bounds=[1,100], item_subjective_ratio_bounds=[0.8,1.2],
    normalized_sum_of_values=1000)
```



Try various algorithms:

```python
print(divide(fairpyx.algorithms.round_robin, instance=random_instance))
print(divide(fairpyx.algorithms.bidirectional_round_robin, instance=random_instance))
print(divide(fairpyx.algorithms.serial_dictatorship, instance=random_instance))
print(divide(fairpyx.algorithms.almost_egalitarian_with_donation, instance=random_instance))
```

```
{'s1': ['c1', 'c2'], 's2': ['c1', 'c2'], 's3': ['c2', 'c3'], 's4':
['c2', 'c3'], 's5': ['c2', 'c3'], 's6': ['c2', 'c3'], 's7': ['c1',
'c3'], 's8': ['c1'], 's9': ['c1'], 's10': ['c1']}
{'s1': ['c2'], 's2': ['c2'], 's3': ['c2'], 's4': ['c2', 'c3'], 's5':
['c1', 'c2'], 's6': ['c1', 'c2'], 's7': ['c1', 'c3'], 's8': ['c1',
'c3'], 's9': ['c1', 'c3'], 's10': ['c1', 'c3']}
{'s1': ['c1', 'c2', 'c3'], 's2': ['c1', 'c2'], 's3': ['c1', 'c2'],
's4': ['c1', 'c2'], 's5': ['c1', 'c2'], 's6': ['c1', 'c2'], 's7':
['c3'], 's8': ['c3'], 's9': ['c3'], 's10': ['c3']}
{'s1': ['c1', 'c3'], 's2': ['c1'], 's3': ['c2'], 's4': ['c2', 'c3'],
's5': ['c1', 'c2'], 's6': ['c1', 'c3'], 's7': ['c2', 'c3'], 's8':
['c1', 'c3'], 's9': ['c2'], 's10': ['c1', 'c2']}
```


---
Markdown generated automatically from [courses.py](courses.py) using [Pweave](http://mpastell.com/pweave) 0.30.3 on 2025-01-26.
