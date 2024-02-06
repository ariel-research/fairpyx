from mip import *
#
# k = [2, 3]  # The number of different budgets for each student (number of fragments)
# # k can b a different number for each student (and to be an array)
# n = 2  # number of students
# m = 3  # number of courses
# # parameters:
# eps = 2
# delta = 0.5
# t = 2
#
# capacities = [1, 1, 2]
# utilities = [[5, 4, 1], [4, 6, 3]]
# prices = [1.5, 2, 0]
# b0i = [5, 4]
# required = [2, 2]  # how much courses every student need to take
#
# # partition the segment
# budget1 = [3, 6]
# budget2 = [2, 4, 6]
#
# a = [[[1, 0, 1], [1, 1, 0]],  # {0,2}
#      [[0, 1, 1], [1, 1, 0], [1, 1, 0]]]  # {1,2}
#
# model = Model("allocations")
# # decision variables:
# # binary variables indicating if student i got bundle l
# # x = [[model.add_var(var_type=BINARY) for l in range(k)] for i in range(n)] # k[i]
# x = [[model.add_var(var_type=BINARY) for _ in range(k[i])] for i in range(n)]
#
# # excess demand function for each course
# z = [model.add_var(var_type=CONTINUOUS, lb=-capacities[j]) for j in range(m)]
# y = [model.add_var(var_type=CONTINUOUS) for j in range(m)]
#
# # objective function: minimize the excess demand function z_j(u, c, p, b)
# objective_expr = xsum(y[j] for j in range(m))
# model.objective = minimize(objective_expr)
#
# # add two constraints that make sure z_j is absolute value
# # model += y_j  >= z_j
# # model += y_j  >= - z_j  # ||1
# for j in range(m):
#     model += y[j] >= z[j]
#     model += y[j] >= -z[j]
#
# # constraint 1: ∑︁  ∑︁(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) = 𝑐_𝑗 + 𝑧_𝑗  ∀𝑗 ∈ [𝑚], 𝑝_𝑗 > 0
# #            𝑖∈[𝑛] ℓ ∈ [𝑘_𝑖]
# for j in range(m):
#     if prices[j] > 0:
#         model += xsum(x[i][l] * a[i][l][j] for i in range(n)
#                       for l in range(k[i])) == capacities[j] + z[j]
#     else:
#         # constraint 2: ∑     ∑︁(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) ≤ 𝑐𝑗 + 𝑧𝑗 ∀𝑗 ∈ [𝑚], 𝑝𝑗 = 0
#         #  𝑖∈[𝑛] ℓ∈[𝑘_𝑖]
#         model += xsum(x[i][l] * a[i][l][j] for i in range(n)
#                       for l in range(k[i])) <= capacities[j] + z[j]
# # constraint 3: ∑︁𝑥_𝑖ℓ = 1  ∀𝑖 ∈ [𝑛]
# #               ℓ∈[𝑘_𝑖]
# for i in range(n):
#     model += xsum(x[i][l] for l in range(k[i])) == 1
#
# # TODO: add EF-TB constraints
# # model += x[0][0] + x[1][0] <= 1
# # model += x[0][0] + x[1][1] <= 1
# # model += x[0][0] + x[1][2] <= 1
# # model += x[0][1] + x[1][0] <= 1
# model += x[0][1] + x[1][1] <= 1
# # model += x[0][1] + x[1][2] <= 1
#
#
#
# # optimizing
# model.optimize()
#
# # Check if the optimization was successful
# if model.num_solutions:
#     # Print the objective value (minimum excess demand function)
#     print("Objective Value:", model.objective_value)
#
#     # Print the values of decision variables x_il
#     for i in range(n):
#         for l in range(k[i]):
#             print(f"x_{i}{l} =", x[i][l].x)
#
#     # Print the values of decision variables z_j and y_j
#     for j in range(m):
#         # print(f"z_{j} =", z[j].x)
#         print(f"|z_{j}|=y_{j} =", y[j].x)
#
# else:
#     print("Optimization was not successful. Status:", model.status)


def optimize_model(different_budgets, capacities, prices, t):
    model = Model("allocations")
    n = len(different_budgets)  # Number of students
    m = len(capacities)  # Number of courses
    k = [len(budgets) for budgets in different_budgets]  # Number of bundles for each student

    # Decision variables
    x = [[model.add_var(var_type=BINARY) for _ in range(k[i])] for i in range(n)]
    z = [model.add_var(var_type=CONTINUOUS, lb=-capacities[j]) for j in range(m)]
    y = [model.add_var(var_type=CONTINUOUS) for j in range(m)]

    # Objective function
    objective_expr = xsum(y[j] for j in range(m))
    model.objective = minimize(objective_expr)

    # Add constraints for absolute value of excess demand
    for j in range(m):
        model += y[j] >= z[j]
        model += y[j] >= -z[j]

    # Course allocation constraints
    for j in range(m):
        if prices[j] > 0:
            model += xsum(x[i][l] * different_budgets[i][l][j] for i in range(n) for l in range(k[i])) == capacities[j] + z[j]
        else:
            model += xsum(x[i][l] * different_budgets[i][l][j] for i in range(n) for l in range(k[i])) <= capacities[j] + z[j]

    # One bundle per student constraint
    for i in range(n):
        model += xsum(x[i][l] for l in range(k[i])) == 1

    # Add EF-TB constraints based on parameter t
    if t == "NO_EF_TB":
        pass  # No EF-TB constraints
    elif t == "EF_TB":
        # Add EF-TB constraints here
        pass
    elif t == "CONTESTED_EF_TB":
        # Add contested EF-TB constraints here
        pass

    # Optimize the model
    model.optimize()

    # Process and print results
    if model.num_solutions:
        print("Objective Value:", model.objective_value)
        for i in range(n):
            for l in range(k[i]):
                print(f"x_{i}{l} =", x[i][l].x)
        for j in range(m):
            print(f"|z_{j}|=y_{j} =", y[j].x)
    else:
        print("Optimization was not successful. Status:", model.status)

# Example usage:
# optimize_model(different_budgets, capacities, prices, t)
