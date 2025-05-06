"""

Article: Repeated Fair Allocation of Indivisible Items by Igarashi, Lackner, Nardi, Novaro
Link: https://arxiv.org/pdf/2304.01644
 
This module implements 3 main versions of the algorithm:
 1. Two agents, two rounds
 2. Two agents, even number of rounds
 3. More than two agents (cyclic algorithm)

Programmer: Shaked Shvartz
Since: 2025-05
"""
from itertools import cycle
from fairpyx import Instance, AllocationBuilder
import logging

logger = logging.getLogger(__name__)


def two_agents_two_rounds(alloc: AllocationBuilder):
    """
    מיישם את הגרסה של שני סוכנים (n=2) ושני סבבים (k=2).
    מתחיל מ-EF+PO גלובלי, מפרק ל-I1, I2, O, ומבצע העברות על O עד EF1 בכל סבב.

    This is the simplest version of the algorithm, where two agents are allocated items in two rounds.
    It satisfies the EF1 condition in each round, and the allocation is done in a way that ensures fairness.
    The algorithm starts with a global EF+PO allocation, then splits the items into two sets: I1 and I2, and the remaining items O.
    
    דוגמה רצה:


    example allocation (n=2, k=2):    
    agents=A,B; items={a,b}; values are reversed for each agent.
      A: u(a)=4,u(b)=1
      B: u(a)=2,u(b)=3

    output:
      1: A→{a},B→{b}
      2: A→{b},B→{a}
    """
    logger.info("Running 2-agents, 2-rounds allocation")
    raise NotImplementedError


def two_agents_even_rounds(alloc: AllocationBuilder, k: int):
    """
    מיישם את הגרסה של שני סוכנים (n=2) עם מספר סבבים זוגי k.
    מבצע חזרה על הבלוק EF בגודל 2, k/2 פעמים.
    
    This version of the algorithm is designed for two agents and an even number of rounds (k).
    It uses a block of EF allocation of size 2, and repeats it k/2 times.
    By doing so, it ensures that the allocation is fair and satisfies the EF1 condition in each round.

    דוגמה רצה (n=2, k=4):

    example allocation (n=2, k=4):
    agents=A,B; items={x,y}; values are reversed for each agent.
      A: u(x)=3,u(y)=1
      B: u(x)=1,u(y)=3

    output:
      1: A→{x},B→{y}
      2: A→{y},B→{x}
      3: A→{x},B→{y}
      4: A→{y},B→{x}
    """
    logger.info("Running 2-agents, even-rounds allocation (k=%d)", k)
    raise NotImplementedError


def multi_agent_cyclic(alloc: AllocationBuilder, k: int):
    """
    מיישם את הגרסה לכל n>2 ו-k כפולה של n
    'אלגוריתם הסיבוב': מחלק בסיס ל-A1..An, ואז cyclic shift על פני k סבבים.
    
    This version of the algorithm is designed for more than two agents and when k is a multiple of n.
    the algorithm works by dividing the items into n sets, one for each agent, and then performing a cyclic shift over k rounds.
    This ensures that each agent gets a fair share of the items in each round.

    דוגמה רצה (n=3, k=3):
    סוכנים=1,2,3; פריטים={a,b,c}; ערכים שווים לכל סוכן.
    
    example allocation (n=3, k=3):
    agents=1,2,3; items={a,b,c}; values are equal for all agents.
      1: u(a)=1,u(b)=1,u(c)=1
      2: u(a)=1,u(b)=1,u(c)=1
      3: u(a)=1,u(b)=1,u(c)=1

    output:
      1: 1→{a},2→{b},3→{c}
      2: 1→{c},2→{a},3→{b}
      3: 1→{b},2→{c},3→{a}
    """
    logger.info("Running multi-agent cyclic allocation (k=%d)", k)
    raise NotImplementedError


if __name__ == "__main__":
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    pass
