"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish, Gérard P. Cachon, Judd B. Kessler, Abraham Othman
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
import logging
import random
import time
import numpy as np
from fairpyx import Instance, AllocationBuilder
from itertools import combinations


# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


"""
Algorithm 1: Approximate Competitive Equilibrium from Equal Incomes (A-CEEI), finds the best price vector that matches student preferences and course capacities.
"""
def A_CEEI(alloc: AllocationBuilder, budget : dict , time_limit: int = 60, seed = None) -> dict:
    """
    Perform heuristic search to find the best price vector that matches student preferences and course capacities.

    :param allocation: Allocation object.
    :param budget (float): Initial budget.
    :param time (float): Time limit for the search.

    :return (dict) best price vector.
    
    :example
    
    >>> instance = Instance(
    ... agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ... item_conflicts = {"c1": [], "c2": [], "c3": []},
    ... agent_capacities = {"Alice": 1, "Bob": 1, "Tom": 1}, 
    ... item_capacities  = {"c1": 1, "c2": 1, "c3": 1},
    ... valuations = {"Alice": {"c1": 100, "c2": 0, "c3": 0},
    ...         "Bob": {"c1": 0, "c2": 100, "c3": 0},
    ...         "Tom": {"c1": 0, "c2": 0, "c3": 100}
    ... })
    >>> budget = {"Alice": 1.0, "Bob": 1.1, "Tom": 1.3}    
    >>> allocation = AllocationBuilder(instance)
    >>> {k: round(v) for k, v in A_CEEI(allocation, budget, 10, 60).items()}
    {'c1': 1, 'c2': 1, 'c3': 1}


    """
    logger.info("Starting A_CEEI algorithm with budget=%s and time limit %d.",budget, time_limit)
   
    def initialize_price_vector(budget,seed):
        return {k: random.uniform(0, max(budget.values())) for k in alloc.instance.items}
    
    best_error = float('inf')
    best_price_vector = dict()
    start_time = time.time()
    steps = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example step sizes, can be adjusted

    preferred_schedule = find_preferred_schedule_adapter(alloc)
    logger.debug("Calling find_preference_order_for_each_student %s",preferred_schedule)
    while time.time() - start_time < time_limit :
        if seed:
            seed+=1        
            random.seed(seed)
        price_vector = initialize_price_vector(budget,seed)

        search_error = alpha(compute_surplus_demand_for_each_course(price_vector, alloc, budget, preferred_schedule))
        logger.debug("Initial search on _random_ price_ %s error: %f",price_vector, search_error)

        tabu_list = []
        c = 0
        while c < 5:
            neighbors = find_neighbors(price_vector, alloc, budget, steps, preferred_schedule)
            logger.debug("Found %d neighbors : %s", len(neighbors), neighbors)
            
            while neighbors:
                next_price_vector = neighbors.pop(0)
                next_demands = compute_surplus_demand_for_each_course(next_price_vector, alloc, budget, preferred_schedule)

                if next_demands not in tabu_list:
                    break
 
            if not neighbors: #if there are neighbors is empty
                logger.debug("all the demand neighbors in tabu_list, break while c<5, tabu_list: %s", tabu_list)
                c = 5
            else:
                price_vector = next_price_vector
                tabu_list.append(next_demands)
                current_error = alpha(next_demands)
                logger.debug("fond neighber so the next price vector is: %s,demands = %s", next_price_vector,next_demands)
                if current_error < search_error:
                    logger.debug("Current error: %f, < Search error: %f soo c = 0", current_error, search_error)

                    search_error = current_error
                    c = 0
                else:
                    c += 1
                    logger.debug("Current error: %f, => Search error: %f soo c = %d", current_error, search_error,c)

                if current_error < best_error:
                    logger.info("New best_price_vector is %s, best error: %f ", price_vector, current_error)
                    best_error = current_error
                    best_price_vector = price_vector
                    if best_error == 0:
                        break
    logger.info("A-CEEI algorithm completed. Best price vector: %s with error: %f", best_price_vector, best_error)      
    return best_price_vector



def find_preference_order_for_each_student(valuations:dict, agent_capacities:dict, item_conflicts:dict, agent_conflicts:dict):
    """
    Finds, for each student, the complete preference ordering on all possible schedules.
    This is a pre-processing step: we compute all preference ordering once, and then use it to find the best schedule that fits the budget.
    
    :param valuations: Dictionary of valuations.
    :param agent_capacities: Dictionary of agent capacities.
    :param item_conflicts: Dictionary of item conflicts.
    :param agent_conflicts: Dictionary of agent conflicts.

    :return (dict) Dictionary of preferred schedules.

    :example
    >>> agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 1}
    >>> item_conflicts = {"c1": ["c2"], "c2": ["c1"], "c3": []}
    >>> agent_conflicts = {"Alice": ["c2"], "Bob": [], "Tom": []}
    >>> valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50}, "Bob": {"c1": 50, "c2": 81, "c3": 60}, "Tom": {"c1": 100, "c2": 95, "c3": 30}}
    >>> find_preference_order_for_each_student(valuations, agent_capacities, item_conflicts, agent_conflicts)
    {'Alice': [[1, 0, 1], [1, 0, 0], [0, 0, 1]], 'Bob': [[0, 1, 1], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]], 'Tom': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
    """
        
    def is_valid_schedule(schedule, item_conflicts, agent_conflicts, agent):
        '''
        Check if a schedule is valid.
        :param schedule: Dictionary of items in the schedule.
        :param item_conflicts: Dictionary of item conflicts.
        :param agent_conflicts: Dictionary of agent conflicts.
        :param agent: Agent name.
        
        :return (bool) True if the schedule is valid, False otherwise.
        
        :example
        >>> schedule = {"c1": 1, "c2": 0, "c3": 1}
        >>> item_conflicts = {"c1": ["c2"], "c2": ["c1"], "c3": []}
        >>> agent_conflicts = {"Alice": ["c2"], "Bob": [], "Tom": []}
        >>> is_valid_schedule({"c1": 1, "c2": 0, "c3": 1}, item_conflicts, agent_conflicts, "Alice")
        True
        >>> is_valid_schedule({"c1": 1, "c2": 1, "c3": 1}, item_conflicts, agent_conflicts, "Alice")
        False
        '''
        # Check item conflicts
        for item, conflicts in item_conflicts.items():
            if schedule.get(item, 0) == 1:
                for conflicted_item in conflicts:
                    if schedule.get(conflicted_item, 0) == 1:
                        return False
        # Check agent conflicts
        for conflicted_item in agent_conflicts.get(agent, []):
            if schedule.get(conflicted_item, 0) == 1:
                return False
        return True


    def generate_all_schedules(items, capacity):
        '''
        Generate all possible schedules for a given set of items and capacity.
        :param items: List of items.
        :param capacity: Capacity.
        
        :return (list) List of all possible schedules.
        
        :example
        >>> items = ["c1", "c2", "c3"]
        >>> capacity = 2
        >>> generate_all_schedules(items, capacity)
        [{'c1': 1, 'c2': 0, 'c3': 0}, {'c1': 0, 'c2': 1, 'c3': 0}, {'c1': 0, 'c2': 0, 'c3': 1}, {'c1': 1, 'c2': 1, 'c3': 0}, {'c1': 1, 'c2': 0, 'c3': 1}, {'c1': 0, 'c2': 1, 'c3': 1}]
        '''
        all_schedules = []
        for num_courses_per_agent in range(1, capacity + 1):
            for schedule in combinations(items, num_courses_per_agent):
                schedule_dict = {item: 1 if item in schedule else 0 for item in items}
                if is_valid_schedule(schedule_dict, item_conflicts, agent_conflicts, agent):
                    all_schedules.append(schedule_dict)
        return all_schedules
    

    preferred_schedules = dict()

    for agent in agent_capacities.keys():
        items = valuations[agent].keys()
        capacity = agent_capacities[agent]
        all_schedules = generate_all_schedules(items, capacity)
        logger.debug("All schedules for agent %s: %s", agent, all_schedules)
        # Calculate valuations for valid schedules
        schedule_valuations = dict()
        for schedule in all_schedules:
            total_valuation = sum(valuations[agent][item] for item in schedule if schedule[item] == 1)
            schedule_valuations[total_valuation] = schedule_valuations.get(total_valuation, [])
            schedule_valuations[total_valuation].append([schedule[item] for item in items])
        # Sort the schedules by total valuation in descending order
        sorted_valuations = sorted(schedule_valuations.keys(), reverse=True)

        for val in sorted_valuations:
            if len(schedule_valuations.get(val)) > 1:
                schedule_valuations[val] = sorted(schedule_valuations.get(val), key=lambda x: sum(x), reverse=True)

        # Collect sorted schedules
        sorted_schedules = []

        for val in sorted_valuations:
            for schedule in schedule_valuations.get(val):
                sorted_schedules.append(schedule)

        preferred_schedules[agent] = sorted_schedules
    return preferred_schedules



def compute_surplus_demand_for_each_course(price_vector: dict ,alloc: AllocationBuilder,  budget : dict, preferred_schedule: dict):
    """
    :param price_vector: List of prices.
    :param allocation: Allocation object.
    :param budget: Dictionary of budgets.
    :param preferred_schedule: Dictionary that maps each student to his preference order on schedules.

    :return (dict) Dictionary of course demands.

    :example
    >>> instance = Instance(
    ...   agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ...   item_conflicts = {"c1": [], "c2": [], "c3": []},
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2},
    ...   valuations = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                 "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                 "Tom": {"c1": 70, "c2": 30, "c3": 70}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {'Alice': [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]], 
    ...                     'Bob': [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 
    ...                     'Tom': [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]}
    >>> compute_surplus_demand_for_each_course(price_vector,allocation , budget, preferred_schedule)
    {'c1': 2, 'c2': -1, 'c3': 0}
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
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice": [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> find_best_schedule(price_vector, budget, preferred_schedule) #    {"Alice":  "AC", "Bob": "AB", "Tom": "AC"}
    [[1, 0, 1], [1, 1, 0], [1, 0, 1]]


    >>> price_vector = {'c1': 1.2, 'c2': 0.9, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {"Alice": [[1, 0, 1], [0, 1, 1], [1, 1, 0]] , "Bob": [[1, 1, 0], [1, 0, 1], [0, 1, 1]], "Tom": [[1, 0, 1], [1, 1, 0], [0, 1, 1]]}
    >>> find_best_schedule(price_vector, budget, preferred_schedule) #    {"Alice":  "BC", "Bob": "AB", "Tom": "AC"}
    [[0, 1, 1], [1, 1, 0], [1, 0, 1]]

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
    result = np.sqrt(sum([v**2 for v in demands.values()]))
    return result


def find_neighbors(price_vector: dict ,alloc: AllocationBuilder, budget : dict, steps: list, preferred_schedule: dict):    
    """
    :param price_vector: List of prices.
    :param allocation: Allocation object.
    :param budget: Dictionary of budgets.
    :param steps: List of steps.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (list of list) List of neighbors.

    :example
    >>> instance = Instance(
    ...   agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ...   item_conflicts = {"c1": [], "c2": [], "c3": []},
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2},
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 50, "c2": 20, "c3": 80},
    ...                 "Bob": {"c1": 60, "c2": 40, "c3": 30},
    ...                 "Tom": {"c1": 70, "c2": 30, "c3": 70}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> steps = [0.1, 0.2]
    >>> preferred_schedule = {'Alice': [[1, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 0], [1, 0, 0], [0, 1, 0]], 
    ...                     'Bob': [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 
    ...                     'Tom': [[1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]}
    >>> find_neighbors(price_vector, allocation, budget, steps, preferred_schedule)
    [{'c1': 1.2, 'c2': 0.9, 'c3': 1.0}, {'c1': 1.4, 'c2': 0.8, 'c3': 1.0}, {'c1': 1.1, 'c2': 1.0, 'c3': 1.0}, {'c1': 1.0, 'c2': 0.0, 'c3': 1.0}]

    """
    demands = compute_surplus_demand_for_each_course(price_vector, alloc, budget, preferred_schedule)
    list_of_neighbors = generate_gradient_neighbors(price_vector, demands, steps)
    list_of_neighbors.extend(generate_individual_adjustment_neighbors(price_vector, alloc, demands, budget, preferred_schedule))

    #sort list_of_neighbors dict values by alpha
    sorted_neighbors = sorted(list_of_neighbors, key=lambda neighbor: alpha(compute_surplus_demand_for_each_course(neighbor, alloc, budget, preferred_schedule)))
    return sorted_neighbors


def generate_individual_adjustment_neighbors(price_vector: dict, alloc: AllocationBuilder, demands: dict, budget : dict , preferred_schedule: dict):
    """
    Generate individual adjustment neighbors.

    :param price_vector: List of prices.
    :param allocation: Allocation object.
    :param demands: Dictionary of course demands.
    :param budget: Dictionary of budgets.
    :param preferred_schedule: Dictionary of preferred schedules.

    :return (list of list) List of individual adjustment neighbors.

    :example
    >>> instance = Instance(
    ...   agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ...   item_conflicts = {"c1": [], "c2": [], "c3": []},
    ...   agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 2},
    ...   item_capacities  = {"c1": 1, "c2": 2, "c3": 2},
    ...   valuations = {"Alice": {"c1": 90, "c2": 60, "c3": 50},
    ...                 "Bob": {"c1": 57, "c2": 80, "c3": 63},
    ...                 "Tom": {"c1": 70, "c2": 50, "c3": 95}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> budget = {"Alice": 2.0, "Bob": 2.1, "Tom": 2.3}
    >>> preferred_schedule = {'Alice': [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 
    ...                     'Bob': [[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 0, 0]], 
    ...                     'Tom': [[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]}
    >>> demands = {'c1': 2, 'c2': -1, 'c3': 0}
    >>> generate_individual_adjustment_neighbors(price_vector, allocation, demands, budget, preferred_schedule)
    [{'c1': 1.1, 'c2': 1.0, 'c3': 1.0}, {'c1': 1.0, 'c2': 0.0, 'c3': 1.0}]

    """
    step=0.1
    neighbors = []
    for k in demands.keys():
        if demands[k] == 0:
            continue
        new_price_vector = price_vector.copy()
        new_demands = demands.copy()
        count=0
        if demands[k] > 0:
            while (demands == new_demands) :
                new_price_vector.update({k: new_price_vector[k] + step})
                new_demands = compute_surplus_demand_for_each_course(new_price_vector, alloc, budget, preferred_schedule)
                count+=1
        elif demands[k] < 0:
            new_price_vector.update({k: 0.0})
            new_demands = compute_surplus_demand_for_each_course(new_price_vector, alloc, budget, preferred_schedule)
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
    [{'c1': 0.2, 'c2': 0.0, 'c3': 0.0}, {'c1': 0.4, 'c2': 0.0, 'c3': 0.0}]

    >>> price_vector = {'c1': 1.0, 'c2': 1.0, 'c3': 1.0}
    >>> demands = {"c1": 2, "c2": -1, "c3": 0}
    >>> steps = [0.1, 0.2, 0.3, 0.4, 0.5]
    >>> generate_gradient_neighbors(price_vector, demands, steps)
    [{'c1': 1.2, 'c2': 0.9, 'c3': 1.0}, {'c1': 1.4, 'c2': 0.8, 'c3': 1.0}, {'c1': 1.6, 'c2': 0.7, 'c3': 1.0}, {'c1': 1.8, 'c2': 0.6, 'c3': 1.0}, {'c1': 2.0, 'c2': 0.5, 'c3': 1.0}]
    """
    neighbors = []
    for step in steps:
        new_price_vector = dict()
        for k,p in price_vector.items():
            new_price_vector[k] = max(0.0, p + (step * demands[k]))

        neighbors.append(new_price_vector)
    return neighbors  

def find_preferred_schedule_adapter(alloc: AllocationBuilder):
    '''
    Helper function to arrange data for the "find_preference_order_for_each_student" function
    
    :param alloc: Allocation object.
    
    :return (dict) Dictionary of preferred schedules.
    
    >>> instance = Instance(
    ... agent_conflicts = {"Alice": [], "Bob": [], "Tom": []},
    ... item_conflicts = {"c1": [], "c2": [], "c3": []},
    ... agent_capacities = {"Alice": 2, "Bob": 2, "Tom": 1},
    ... item_capacities  = {"c1": 1, "c2": 1, "c3": 1},
    ... valuations = {"Alice": {"c1": 100, "c2": 0, "c3": 0},
    ...         "Bob": {"c1": 0, "c2": 100, "c3": 0},
    ...         "Tom": {"c1": 0, "c2": 0, "c3": 100}
    ... })
    >>> allocation = AllocationBuilder(instance)
    >>> find_preferred_schedule_adapter(allocation)
    {'Alice': [[1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1]], 'Bob': [[1, 1, 0], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0], [0, 0, 1]], 'Tom': [[0, 0, 1], [1, 0, 0], [0, 1, 0]]}
    '''
    item_conflicts={item:  alloc.instance.item_conflicts(item) for item in alloc.instance.items}
    agent_conflicts={agent:  alloc.instance.agent_conflicts(agent) for agent in alloc.instance.agents}
    return find_preference_order_for_each_student(alloc.instance._valuations , alloc.instance._agent_capacities , item_conflicts , agent_conflicts)

if __name__ == "__main__":
    import doctest
    print(doctest.testmod())

