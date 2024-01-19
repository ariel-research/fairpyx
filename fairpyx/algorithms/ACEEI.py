import logging
from fairpyx import Instance

logger = logging.getLogger(__name__)


def find_ACEEI_with_EFTB(instance: Instance, initial_budgets, delta, epsilon, t):
    """
    find an ACEEI with (contested) EF-TB
    :param instance: a fair-course-allocation instance
    :param initial_budgets: Students' initial budgets
    :param delta: The step size
    :param epsilon: maximum budget perturbation
    :param t: type 𝑡 of the EF-TB constraint,
              0 for no EF-TB constraint,
              1 for EF-TB constraint,
              2 for contested EF-TB
    :return finial courses prices, finial budgets, finial distribution

     >>> from fairpyx.adaptors import divide

    >>> from fairpyx.utils.test_utils import stringify

    >>> instance = Instance(valuations={"avi":{"x":1, "y":2, "z":4}, "beni":{"x":2, "y":3, "z":1}},agent_capacities=2,item_capacities={1, 1, 2})
    >>> initial_budgets = {2, 3}
    >>> delta = 0.5
    >>> epsilon = 0.5
    >>> t = 0 #TODO: check if we put the arguments in the right way
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,*initial_budgets, *delta, *epsilon, *t))
    "{p:{1, 2, 0}, b:{1.5, 2.5}, allocation:{avi:['x','z'], beni:['y', 'z']}}"

    >>> instance = Instance(valuations={"avi":{"x":5, "y":2, "z":1}, "beni":{"x":4, "y":1, "z":3}},agent_capacities=2,item_capacities={1, 1, 2})
    >>> initial_budgets = {3, 4}
    >>> delta = 0.5
    >>> epsilon = 1
    >>> t = 1
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,*initial_budgets, *delta, *epsilon, *t))
    "{p:{2.5, 0, 0}, b:{2, 4}, allocation:{avi:['y','z'], beni:['x', 'z']}}"

    >>> instance = Instance(valuations={"avi":{"x":5, "y":5, "z":1}, "beni":{"x":4, "y":6, "z":4}},agent_capacities=2,item_capacities={1, 2, 2})
    >>> initial_budgets = {5, 4}
    >>> delta = 0.5
    >>> epsilon = 2
    >>> t = 1
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,*initial_budgets, *delta, *epsilon, *t))
    "{p:{2.5, 0, 0}, b:{5, 2}, allocation:{avi:['x','y'], beni:['y', 'z']}}"

    >>> instance = Instance(valuations={"avi":{"x":10, "y":20}, "beni":{"x":10, "y":20}},agent_capacities=1,item_capacities = {1, 1})
    >>> initial_budgets = {1.1, 1}
    >>> delta = 0.1
    >>> epsilon = 0.2
    >>> t = 1
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,*initial_budgets, *delta, *epsilon, *t))
    "{p:{0, 0.9}, b:{1.1, 0.8}, allocation:{avi:['y'], beni:['x']}}"

    >>> instance = Instance(valuations={"avi":{"x":2}, "beni":{"x":3}},agent_capacities=1,item_capacities = {1})
    >>> initial_budgets = {1.1, 1}
    >>> delta = 0.1
    >>> epsilon = 0.2
    >>> t = 1
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,*initial_budgets, *delta, *epsilon, *t))
    "{p:{0.9}, b:{1.1, 0.8}, allocation:{avi:['x'], beni:[]}}"

    >>> instance = Instance(valuations={"avi":{"x":5, "y":4, "z":1}, "beni":{"x":4, "y":6, "z":3}},agent_capacities=2,item_capacities={1, 1, 2})
    >>> initial_budgets = {5, 4}
    >>> delta = 0.5
    >>> epsilon = 2
    >>> t = 2
    >>> stringify(divide(find_ACEEI_with_EFTB, instance=instance,*initial_budgets, *delta, *epsilon, *t))
    "{p:{1.5, 2, 0}, b:{3, 2}, allocation:{avi:['x', 'z'], beni:['y', 'z']}}"
    """
