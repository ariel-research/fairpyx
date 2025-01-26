# Demo of picking-sequence algorithms


```python
import fairpyx
from fairpyx.algorithms.picking_sequence import round_robin

import logging
round_robin.logger.addHandler(logging.StreamHandler())
round_robin.logger.setLevel(logging.INFO)

# The preference rating of the courses for each of the students:
valuations = {
    "Alice": {"c1":2, "c2": 3, "c3": 4},
    "Bob": {"c1": 4, "c2": 5, "c3": 6}
}
agent_capacities = {"Alice": 2, "Bob": 1}
course_capacities = {"c1": 2, "c2": 1, "c3": 1}

instance = fairpyx.Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
print(fairpyx.divide(round_robin, instance=instance))
```

```

Picking-sequence with items {'c1': 2, 'c2': 1, 'c3': 1} , agents
{'Alice': 2, 'Bob': 1}, and agent-order ['Alice', 'Bob']
Agent Alice takes item c3 with value 4
Agent Bob takes item c2 with value 5
Agent Alice takes item c1 with value 2
No more items to allocate
{'Alice': ['c1', 'c3'], 'Bob': ['c2']}
```


---
Markdown generated automatically from [picking_sequence.py](picking_sequence.py) using [Pweave](http://mpastell.com/pweave) 0.30.3 on 2025-01-26.
