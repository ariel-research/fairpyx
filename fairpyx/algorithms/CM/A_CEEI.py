"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
import random
import time
import numpy as np
from fairpyx.instances import Instance
from fairpyx import Instance, AllocationBuilder, divide

"""
Algorithm 1: Approximate Competitive Equilibrium from Equal Incomes (A-CEEI), finds the best price vector that matches student preferences and course capacities.
"""
def A_CEEI(alloc: AllocationBuilder, budget : dict , time_limit: int = 60):
    """
    Perform heuristic search to find the best price vector that matches student preferences and course capacities.

    :param instance (Instance): Instance object.
    :param budget (float): Initial budget.
    :param time (float): Time limit for the search.

    :return (dict) best price vector.

    :example
    #>>> instance = Instance(
    #...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2},
    #...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    #...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    #...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    #...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    #... })
    #>>> allocation = AllocationBuilder(instance)
    #>>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    #>>> A_CEEI(allocation, budget, 2)
    #{'c1': 1.4896292402060682, 'c2': 0.6800500563258562, 'c3': 1.33237205943072}

    """
    def initialize_price_vector(budget):
        return {k: random.uniform(0, max(budget.values())) for k in alloc.instance.items}
    
    best_error = float('inf')
    best_price_vector = None
    start_time = time.time()
    steps = [0.1, 0.2, 0.3]  # Example step sizes, can be adjusted
    preferred_schedule = find_preferred_schedule(alloc)
    
    while time.time() - start_time < time_limit:
        price_vector = initialize_price_vector(budget)
        search_error = float('inf')
        tabu_list = []
        c = 0
        
        while c < 5:
            neighbors = find_neighbors(price_vector, alloc, budget, steps, preferred_schedule)
            found_next_step = False
            
            while neighbors:
                next_price_vector = neighbors.pop(0)
                next_demands = course_demands(next_price_vector, alloc, budget, preferred_schedule)
                
                if next_demands not in tabu_list:
                    found_next_step = True
                    break
            
            if not found_next_step:
                c = 5
            else:
                price_vector = next_price_vector
                tabu_list.append(next_demands)
                current_error = alpha(next_demands)
                
                if current_error < search_error:
                    search_error = current_error
                    c = 0
                else:
                    c += 1
                
                if current_error < best_error:
                    best_error = current_error
                    best_price_vector = price_vector
    
    return best_price_vector

def find_preferred_schedule(alloc: AllocationBuilder):
    """
    Find the preferred schedule for each student.

    :param instance: Instance object.

    :return (dict) Dictionary of preferred schedules.
    """
    return {"Alice":  [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}


def course_demands(price_vector: dict ,alloc: AllocationBuilder,  budget : dict, preferred_schedule: dict):
    """
    :param price_vector: List of prices.
    :param instance: Instance object.
    :param budget: Dictionary of budgets.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (dict) Dictionary of course demands.

    :example
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice":  [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> course_demands(price_vector,allocation , budget, preferred_schedule)
    {'c1': 2, 'c2': -1, 'c3': 0}

    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.2, 'c2': 0.9, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice":  [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> course_demands(price_vector, allocation, budget, preferred_schedule)
    {'c1': 1, 'c2': 0, 'c3': 0}
    """
    best_schedules = find_best_schedule(price_vector, budget, preferred_schedule)
    sol = np.sum(np.array(best_schedules), axis=0)
    # Convert item capacities to a list
    item_capacities_list = [alloc.instance.item_capacity(name_item) for name_item in alloc.instance.items]

    # Convert item capacities list to a numpy array
    item_capacities_array = np.array(item_capacities_list)

    # Perform the subtraction
    result = {name_item: int(sol[i] - item_capacities_array[i]) for i, name_item in enumerate(alloc.instance.items)}
    return result
       

def find_best_schedule(price_vector: dict, budget : dict, preferred_schedule: dict):    
    """
    Find the best schedule for a student considering the price vector and the budget.

    :param price_vector: List of prices.
    :param budget: Dictionary of budgets.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (list) List of courses in the best schedule.

    :example
    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2},
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice": [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> find_best_schedule(price_vector, budget, preferred_schedule) #    {"Alice":  "AC", "Bob": "AB", "Tom": "AC"}
    [[1, 0, 1], [1, 1, 0], [1, 0, 1]]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2}, 
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> price_vector = {'c1': 1.2, 'c2': 0.9, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice": [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> find_best_schedule(price_vector, budget, preferred_schedule) #    {"Alice":  "BC", "Bob": "AB", "Tom": "AC"}
    [[0, 1, 1], [1, 1, 0], [1, 0, 1]]

    >>> instance = Instance(
    ...   agent_capacities = {"Alice": 2, "Bob": 1, "Tom": 1},
    ...   item_capacities  = {"c1": 2, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 100, "c2": 100, "c3": 0},
    ...                 "Bob": {"c1": 0, "c2": 100, "c3": 0},
    ...                 "Tom": {"c1": 0, "c2": 0, "c3": 100}
    ... })
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 1.1, "Tom": 1.3}
    >>> preferred_schedule = {"Alice": [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[0, 1, 0], [0, 0, 1], [1, 0, 0]], "Tom": [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}
    >>> find_best_schedule(price_vector, budget, preferred_schedule) #    {"Alice":  "AB", "Bob": "B", "Tom": "C"}
    [[1, 0, 1], [0, 1, 0], [0, 0, 1]]

    """
    best_schedule = []
    cuont = 0
    price_array = np.array([price_vector[key] for key in price_vector.keys()])
    for student, schedule in preferred_schedule.items():
        best_schedule.append(np.zeros(len(price_vector)))
        sum_of_courses = [i for i in np.sum(schedule * price_array[:, np.newaxis].T, axis=1)]
        for i in range(len(sum_of_courses)):
            if sum_of_courses[i] <= budget[student]:
                best_schedule[cuont] = schedule[i]
                break
        cuont += 1
    
    return best_schedule
               
        
def alpha(demands: dict):
    """
    :param demands: Dictionary of course demands.

    :return (float) Alpha value.

    :example
    >>> demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> alpha(demands) # sqrt(5)
    2.23606797749979
    
    >>> demands = {"c1": 1, "c2": 1, "c3": 1}
    >>> alpha(demands) # sqrt(3)
    1.7320508075688772

    >>> demands = {"c1": 0, "c2": 0, "c3": 0}
    >>> alpha(demands)
    0.0
    """
    return np.sqrt(sum([v**2 for v in demands.values()]))


def find_neighbors(price_vector: dict ,alloc: AllocationBuilder, budget : dict, steps: list, preferred_schedule: dict):    
    """
    :param price_vector: List of prices.
    :param instance: Instance object.
    :param budget: Dictionary of budgets.
    :param steps: List of steps.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (list of list) List of neighbors.

    :example
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> steps = [0.1, 0.2]
    >>> preferred_schedule = {"Alice": [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]} #{"Alice":  ["AC", "CB", "AB"] , "Bob": ["AB", "AC" "BC"], "Tom": ["AC", "AB", "BC"]}
    >>> find_neighbors(price_vector, allocation, budget, steps, preferred_schedule)
    [{'c1': 1.2, 'c2': 0.9, 'c3': 1.0}, {'c1': 1.4, 'c2': 0.8, 'c3': 1.0}, {'c1': 1.1, 'c2': 1.0, 'c3': 1.0}, {'c1': 1.0, 'c2': 0.0, 'c3': 1.0}]

    """

    demands = course_demands(price_vector, alloc, budget, preferred_schedule)
    list_of_neighbors = generate_gradient_neighbors(price_vector, demands, steps)
    list_of_neighbors.extend(generate_individual_adjustment_neighbors(price_vector, alloc, demands, budget, preferred_schedule))

    #sort list_of_neighbors dict values by alpha
    sorted_neighbors = sorted(list_of_neighbors, key=lambda neighbor: alpha(course_demands(neighbor, alloc, budget, preferred_schedule)))
    return sorted_neighbors


def generate_individual_adjustment_neighbors(price_vector: dict, alloc: AllocationBuilder, demands: dict, budget : dict , preferred_schedule: dict):
    """
    Generate individual adjustment neighbors.

    :param price_vector: List of prices.
    :param instance: Instance object.
    :param demands: Dictionary of course demands.
    :param budget: Dictionary of budgets.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (list of list) List of individual adjustment neighbors.

    :example
    >>> instance = Instance(
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice":  [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> demands = {'c1': 2, 'c2': -1, 'c3': 0}
    >>> generate_individual_adjustment_neighbors(price_vector, allocation, demands, budget, preferred_schedule)
    [{'c1': 1.1, 'c2': 1.0, 'c3': 1.0}, {'c1': 1.0, 'c2': 0.0, 'c3': 1.0}]

    """
    neighbors = []

    for k in demands.keys():
        if demands.get(k) == 0:
            continue
        new_price_vector = price_vector.copy()
        new_demands= demands.copy()
        counter=0
        while (demands == new_demands) and counter<100 :
            if demands.get(k) > 0:
                new_price_vector.update({k: new_price_vector.get(k) + 0.1})
            elif demands.get(k) < 0:
                new_price_vector.update({k: 0.0})
                break
            new_demands = course_demands(new_price_vector, alloc, budget, preferred_schedule)
            counter+=1
        neighbors.append(new_price_vector.copy())  # Ensure to append a copy


    return neighbors

def generate_gradient_neighbors(price_vector: dict, demands: dict, steps: list):
    """
    Generate gradient neighbors.

    :param price_vector: List of prices.
    :param demands: Dictionary of course demands.
    :param steps: List of steps.

    :return (list of list) List of gradient neighbors.

    :example
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2]
    >>> generate_gradient_neighbors(price_vector, demands, steps)
    [{'c1': 1.2, 'c2': 0.9, 'c3': 1.0}, {'c1': 1.4, 'c2': 0.8, 'c3': 1.0}]

    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> demands = {"c1": 1, "c2": 1, "c3": 1}
    >>> steps = [0.1, 0.2]
    >>> generate_gradient_neighbors(price_vector, demands, steps)
    [{'c1': 1.1, 'c2': 1.1, 'c3': 1.1}, {'c1': 1.2, 'c2': 1.2, 'c3': 1.2}]

    >>> price_vector = {'c1': 0.0, 'c2': 0.0, 'c3': 0.0}
    >>> demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2]
    >>> generate_gradient_neighbors(price_vector, demands, steps)
    [{'c1': 0.2, 'c2': -0.1, 'c3': 0.0}, {'c1': 0.4, 'c2': -0.2, 'c3': 0.0}]

    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2, 0.3, 0.4, 0.5]
    >>> generate_gradient_neighbors(price_vector, demands, steps)
    [{'c1': 1.2, 'c2': 0.9, 'c3': 1.0}, {'c1': 1.4, 'c2': 0.8, 'c3': 1.0}, {'c1': 1.6, 'c2': 0.7, 'c3': 1.0}, {'c1': 1.8, 'c2': 0.6, 'c3': 1.0}, {'c1': 2.0, 'c2': 0.5, 'c3': 1.0}]
    """
    neighbors = []
    for step in steps:

        new_price_vector = {k: price_vector.get(k) + (step * demands.get(k)) for k in price_vector.keys()}
        neighbors.append(new_price_vector)
    return neighbors  

if __name__ == "__main__":
    import doctest
    doctest.testmod()



