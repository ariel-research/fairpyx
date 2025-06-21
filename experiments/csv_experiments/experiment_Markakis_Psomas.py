import time
import random
import matplotlib.pyplot as plt
from fairpyx import Instance, divide,AllocationBuilder
from fairpyx.algorithms.markakis_psomas import algorithm1_worst_case_allocation
import time
import csv
import numpy as np

# הגדרת אלגוריתמים להשוואה
def round_robin(alloc: AllocationBuilder) -> None:
    agents = list(alloc.remaining_agents())
    items = list(alloc.remaining_items())
    for i, item in enumerate(items):
        agent = agents[i % len(agents)]
        alloc.give(agent, item)

def sequential_allocation(alloc: AllocationBuilder) -> None:
    agents = list(alloc.remaining_agents())
    items = list(alloc.remaining_items())
    for agent in agents:
        if items:
            best_item = max(items, key=lambda item: alloc.effective_value(agent, item))
            alloc.give(agent, best_item)
            items.remove(best_item)

# יצירת קלטים אקראיים
def generate_random_instance(num_agents, num_items, distribution):
    np.random.seed()
    valuations = {}
    
    for i in range(num_agents):
        agent = f"agent{i}"
        if distribution == 'uniform':
            values = np.random.randint(1, 100, num_items)
        elif distribution == 'normal':
            values = np.abs(np.random.normal(50, 20, num_items)).astype(int)
            values = np.where(values < 1, 1, values)
        elif distribution == 'exponential':
            values = np.random.exponential(30, num_items).astype(int)
            values = np.where(values < 1, 1, values)
        
        valuations[agent] = {f"item{j}": float(val) for j, val in enumerate(values)}
    
    return Instance(valuations=valuations)

# הרצת ניסויים
def run_experiments():
    algorithms = {
        "Markakis&Psomas": algorithm1_worst_case_allocation,
        "RoundRobin": round_robin,
        "Sequential": sequential_allocation
    }
    
    num_agents_list = [5, 10, 20, 50]
    num_items_list = [100, 200, 500, 1000]
    distributions = ['uniform', 'normal', 'exponential']
    trials = 5
    
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['algorithm', 'num_agents', 'num_items', 'distribution', 
                      'min_value', 'sum_value', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for num_agents in num_agents_list:
            for num_items in num_items_list:
                if num_items < num_agents:
                    continue
                    
                for distribution in distributions:
                    for trial in range(trials):
                        instance = generate_random_instance(num_agents, num_items, distribution)
                        
                        for algo_name, algo_func in algorithms.items():
                            try:
                                # מדידת זמן ריצה
                                start_time = time.time()
                                
                                # הפעלת האלגוריתם
                                alloc_builder = AllocationBuilder(instance)
                                algo_func(alloc_builder)
                                allocation_dict = alloc_builder.sorted()  # returns a dict
                                
                                end_time = time.time()
                                
                                # חישוב מדדים - ישירות מהמילול
                                min_value = float('inf')
                                sum_value = 0
                                
                                # לכל סוכן, חשב את ערך החבילה שלו
                                for agent in instance.agents:
                                    agent_value = 0
                                    # אם יש פריטים שהוקצו לסוכן
                                    if agent in allocation_dict:
                                        for item in allocation_dict[agent]:
                                            agent_value += instance.agent_item_value(agent, item)
                                    
                                    sum_value += agent_value
                                    if agent_value < min_value:
                                        min_value = agent_value
                                
                                # כתיבת התוצאות
                                writer.writerow({
                                    'algorithm': algo_name,
                                    'num_agents': num_agents,
                                    'num_items': num_items,
                                    'distribution': distribution,
                                    'min_value': min_value,
                                    'sum_value': sum_value,
                                    'execution_time': end_time - start_time
                                })
                                csvfile.flush()
                                print(f"Completed: {algo_name}, agents: {num_agents}, items: {num_items}, dist: {distribution}, trial: {trial}")
                                
                            except Exception as e:
                                print(f"Error in {algo_name} with {num_agents} agents, "
                                      f"{num_items} items, {distribution} distribution: {e}")
                                writer.writerow({
                                    'algorithm': algo_name,
                                    'num_agents': num_agents,
                                    'num_items': num_items,
                                    'distribution': distribution,
                                    'min_value': -1,
                                    'sum_value': -1,
                                    'execution_time': -1
                                })
                                csvfile.flush()

if __name__ == "__main__":
    run_experiments()