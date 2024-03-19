# Input formats


```python
import fairpyx
divide = fairpyx.divide
```



`fairpyx` allows various input formats, so that you can easily use it on your own data,
whether for applications or for research.
For example, suppose you want to divide candies among your children.
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


For experiments, you can use a numpy random matrix:

```python
import numpy as np
valuations = np.random.randint(1,100,[2,4])
print(valuations)
allocation = divide(fairpyx.algorithms.round_robin, valuations=valuations)
print(allocation)
```

```
[[44  1 43 50]
 [83 90 44 20]]
{0: [0, 3], 1: [1, 2]}
```


---
Markdown generated automatically from [input_formats.py](input_formats.py) using [Pweave](http://mpastell.com/pweave) 0.30.3 on 2023-10-18.
