"""
"Course bidding at business schools", by Tayfun Sönmez and M. Utku Ünver (2010)
https://doi.org/10.1111/j.1468-2354.2009.00572.x

Allocate course seats using Gale-Shapley pareto-dominant market mechanism.

Programmer: Zachi Ben Shitrit
Since: 2024-05
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger

import logging
logger = logging.getLogger(__name__)

def gale_shapley(alloc: AllocationBuilder, course_order_per_student: dict, tie_braking_lottery: dict) -> dict:
    """
    Allocate the given items to the given agents using the gale_shapley protocol.
    :param alloc: an allocation builder which tracks agent_capacities, item_capacities, valuations.
    :param course_order_per_student: a dictionary that matches each agent to hes course rankings in order to indicate his preferences
    :param tie_braking_lottery: a dictionary that matches each agent to hes tie breaking additive points (a sample from unified distribution [0,1])

    >>> from fairpyx.adaptors import divide
    >>> s1 = {"c1": 40, "c2": 60}
    >>> s2 = {"c1": 70, "c2": 30}
    >>> s3 = {"c1": 70, "c2": 30}
    >>> s4 = {"c1": 40, "c2": 60}
    >>> s5 = {"c1": 50, "c2": 50}
    >>> agent_capacities = {"Alice": 1, "Bob": 1, "Chana": 1, "Dana": 1, "Dor": 1}
    >>> course_capacities = {"c1": 3, "c2": 2}
    >>> valuations = {"Alice": s1, "Bob": s2, "Chana": s3, "Dana": s4, "Dor": s5}
    >>> course_order_per_student = {"Alice": ["c2", "c1"], "Bob": ["c1", "c2"], "Chana": ["c1", "c2"], "Dana": ["c2", "c1"], "Dor": ["c1", "c2"]}
    >>> tie_braking_lottery = {"Alice": 0.9, "Bob": 0.1, "Chana": 0.2, "Dana": 0.6, "Dor": 0.4}
    >>> instance = Instance(agent_capacities=agent_capacities, item_capacities=course_capacities, valuations=valuations)
    >>> divide(gale_shapley, instance=instance, course_order_per_student=course_order_per_student, tie_braking_lottery=tie_braking_lottery)
    {'Alice': ['c2'], 'Bob': ['c1'], 'Chana': ['c1'], 'Dana': ['c2'], 'Dor': ['c1']}
    """
    
    return {}