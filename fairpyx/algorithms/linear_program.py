from mip import *

def optimize_model(a, capacities, prices, t):
    model = Model("allocations")
    n = len(a)  # Number of students
    courses_names = list(capacities.keys())  # keys of courses
    m = len(courses_names)  # Number of courses
    k = [len(budgets) for budgets in a]  # Number of bundles for each student

    # Decision variables
    x = [[model.add_var(var_type=BINARY) for _ in range(k[i])] for i in range(n)]
    z = [model.add_var(var_type=CONTINUOUS, lb=-capacities[course]) for course in courses_names]
    y = [model.add_var(var_type=CONTINUOUS) for course in range(m)]

    # Objective function
    objective_expr = xsum(y[j] for j in range(m))
    model.objective = minimize(objective_expr)

    # Add constraints for absolute value of excess demand
    for j in range(m):
        model += y[j] >= z[j]
        model += y[j] >= -z[j]

    # Course allocation constraints
    for j, course in enumerate(courses_names):
        # constraint 1: ∑︁  ∑︁(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) = 𝑐_𝑗 + 𝑧_𝑗  ∀𝑗 ∈ [𝑚], 𝑝_𝑗 > 0
        #            𝑖∈[𝑛] ℓ ∈ [𝑘_𝑖]
        if prices[j] > 0:
            model += xsum(x[i][l] * a[i][l][j] for i in range(n) for l in range(k[i])) == capacities[course] + z[j]
        # constraint 2: ∑     ∑︁(𝑥_𝑖ℓ · 𝑎_𝑖ℓ𝑗) ≤ 𝑐𝑗 + 𝑧𝑗 ∀𝑗 ∈ [𝑚], 𝑝𝑗 = 0
        #  𝑖∈[𝑛] ℓ∈[𝑘_𝑖]
        else:
            model += xsum(x[i][l] * a[i][l][j] for i in range(n) for l in range(k[i])) <= capacities[course] + z[j]

    # constraint 3: ∑︁𝑥_𝑖ℓ = 1  ∀𝑖 ∈ [𝑛]
    #               ℓ∈[𝑘_𝑖]
    for i in range(n):
        model += xsum(x[i][l] for l in range(k[i])) == 1

    # # Add EF-TB constraints based on parameter t
    # if t == "NO_EF_TB":
    #     pass  # No EF-TB constraints
    # elif t == "EF_TB":
    #     # Add EF-TB constraints here
    #     pass
    # elif t == "CONTESTED_EF_TB":
    #     # Add contested EF-TB constraints here
    #     pass

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

