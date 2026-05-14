from fairpyx.utils.graph_utils import rank_maximal_matching_algorithm
from fairpyx import Instance, AllocationBuilder

def rank_maximal_matching(alloc: AllocationBuilder):
    instance = alloc.remaining_instance()
    alloc.give_bundles(rank_maximal_matching_algorithm(
        items=instance.items,
        agents=instance.agents,
        agent_item_value=instance.agent_item_value))
    

if __name__ == "__main__":
    import fairpyx   
    import random
    agents  = ["a","b","c","d","e"]
    items   = [1,2,3,4,5]
    valuations = {
            agent: dict(zip(items, [random.randint(1,5) for item in items]
            ))
            for agent in agents
        }
    instance = fairpyx.Instance(valuations=valuations,agents=agents,items=items)
    allocation = fairpyx.divide(fairpyx.algorithms.rank_maximal_matching, instance=instance)
    print(allocation)
    fairpyx.validate_allocation(instance, allocation, title=f"rmm")

