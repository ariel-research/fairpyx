from mip import *

from ACEEI import EFTBStatus


def check_utility(bundle_student, bundle_other_student, student_vals):
    values = list(student_vals.values())
    bundle_student_val = sum(bundle_student * values)
    bundle_other_student_val = sum(bundle_other_student * values)
    return bundle_other_student_val > bundle_student_val


def check_envy(instance, student, idx_student, other_student, idx_other, a):
    courses_names = list(instance._item_capacities.keys())  # keys of courses

    # check if student envies in other_student
    for idx_i, bundle_i in enumerate(a[idx_student]):
        for idx_j, bundle_j in enumerate(a[idx_other]):
            if check_utility(bundle_i, bundle_j, instance._valuations[student]):
                return idx_i, idx_j
    return False, False


def get_envy_constraints(instance, initial_budgets, a, model, x):
    students_names = list(instance._agent_capacities.keys())  # keys of agents

    for idx_student, student in enumerate(students_names):
        for idx_other, other_student in enumerate(students_names):
            if student is not other_student:
                if initial_budgets[idx_student] > initial_budgets[idx_other]:  # check envy
                    # result contain the index of the bundles that student envious other_student
                    result = check_envy(instance, student, idx_student, other_student, idx_other, a)
                    if result:
                        i,j = result
                        model += x[student][i] + x[other_student][j] <=1
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        print(f"student {student} bundle {i} envy student {other_student} bundle {j}")
                        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")





# def optimize_model(a, instance, prices, t, initial_budgets):
#     model = Model("allocations")
#     n = len(a)  # Number of students
#     courses_names = list(instance._item_capacities.keys())  # keys of courses
#     m = len(courses_names)  # Number of courses
#     k = [len(budgets) for budgets in a]  # Number of bundles for each student
#
#     # Decision variables
#     x = [[model.add_var(var_type=BINARY) for _ in range(k[i])] for i in range(n)]
#     z = [model.add_var(var_type=CONTINUOUS, lb=-instance._item_capacities[course]) for course in courses_names]
#     y = [model.add_var(var_type=CONTINUOUS) for course in range(m)]
#
#     # Objective function
#     objective_expr = xsum(y[j] for j in range(m))
#     model.objective = minimize(objective_expr)
#
#     # Add constraints for absolute value of excess demand
#     for j in range(m):
#         model += y[j] >= z[j]
#         model += y[j] >= -z[j]
#
#     # Course allocation constraints
#     for j, course in enumerate(courses_names):
#         # constraint 1: ∑︁  ∑︁(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) = 𝑐_𝑗 + 𝑧_𝑗  ∀𝑗 ∈ [𝑚], 𝑝_𝑗 > 0
#         #            𝑖∈[𝑛] ℓ ∈ [𝑘_𝑖]
#         if prices[j] > 0:
#             model += xsum(x[i][l] * a[i][l][j] for i in range(n) for l in range(k[i])) == instance._item_capacities[
#                 course] + z[j]
#         # constraint 2: ∑     ∑︁(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) ≤ 𝑐𝑗 + 𝑧𝑗 ∀𝑗 ∈ [𝑚], 𝑝𝑗 = 0
#         #  𝑖∈[𝑛] ℓ∈[𝑘_𝑖]
#         else:
#             model += xsum(x[i][l] * a[i][l][j] for i in range(n) for l in range(k[i])) <= instance._item_capacities[
#                 course] + z[j]
#
#     # constraint 3: ∑︁𝑥_𝑖ℓ = 1  ∀𝑖 ∈ [𝑛]
#     #               ℓ∈[𝑘_𝑖]
#     for i in range(n):
#         model += xsum(x[i][l] for l in range(k[i])) == 1
#
#     # Add EF-TB constraints based on parameter t
#     if t == EFTBStatus.NO_EF_TB:
#         pass  # No EF-TB constraints, no need to anything
#     elif t == EFTBStatus.EF_TB:
#         # Add EF-TB constraints here
#         get_envy_constraints(instance, initial_budgets, a, model, x)
#
#     elif t == EFTBStatus.CONTESTED_EF_TB:
#         # Add contested EF-TB constraints here
#         pass
#
#     # Optimize the model
#     model.optimize()
#
#     # Process and print results
#     if model.num_solutions:
#         print("Objective Value:", model.objective_value)
#         for i in range(n):
#             for l in range(k[i]):
#                 print(f"x_{i}{l} =", x[i][l].x)
#         for j in range(m):
#             print(f"|z_{j}|=y_{j} =", y[j].x)
#     else:
#         print("Optimization was not successful. Status:", model.status)

def optimize_model(t):
    print("t= ", t)
