import matplotlib.pyplot as plt
import numpy as np
import math
""" 
    Computes the worst-case guarantee value Vn(alpha) for a given agent.

    This function implements the piecewise definition from Definition 1 in the paper:
    - If alpha == 0 → Vn(alpha) = 1 / n
    - If alpha >= 1 / (n-1) → Vn(alpha) = 0
    - Else → determine k and calculate based on whether alpha ∈ I(n,k) or NI(n,k)

    :param alpha: The value of the largest single item for the agent.
    :param n: The total number of agents.
    :return: The worst-case guaranteed value Vn(alpha).

    -to see how can calculate it k you can see proof for claim 7 in :
     http://pages.cs.aueb.gr/~markakis/research/wine11-Vn.pdf
    -in this class show V(n)-graph
    Programmer: Ibrahem Hurani


"""
import math

def compute_vn(alpha: float, n: int) -> float:
    """ 
    Computes the worst-case guarantee value Vn(alpha) for a given agent.

    This function implements the piecewise definition from Definition 1 in the paper:
    - If alpha == 0 → Vn(alpha) = 1 / n
    - If alpha >= 1 / (n-1) → Vn(alpha) = 0
    - Else → determine k and calculate based on whether alpha ∈ I(n,k) or NI(n,k)

    :param alpha: The value of the largest single item for the agent.
    :param n: The total number of agents.
    :return: The worst-case guaranteed value Vn(alpha).
    """
    if n <= 1:
        return 0.0
    if alpha == 0:
        return 1.0 / n
    
    for k in range(1, 1000):
        I_left=(k + 1) / (k * ((k + 1) * n - 1))
        I_right=1 / (k * n - 1)
        NI_left= 1 / ((k + 1) * n - 1)
        NI_right=(k + 1) / (k * ((k + 1) * n - 1))
        # Check I(n,k) range (closed interval)
        if I_left <= alpha <= I_right or math.isclose(alpha, I_left) or math.isclose(alpha, I_right):
            return 1 - k * (n - 1) * alpha
        # Check NI(n,k) range (open interval)
        if NI_left < alpha < NI_right or math.isclose(alpha, NI_left) or math.isclose(alpha, NI_right):
            return 1 - ((k + 1) * (n - 1)) / ((k + 1) * n - 1)
    return 0.0



def plot_vn():
    alphas = np.linspace(0, 1, 1000)
    plt.figure(figsize=(8, 6))

    for n, color in [(2, 'red'), (3, 'green')]:
        values = [compute_vn(alpha, n) for alpha in alphas]
        plt.plot(alphas, values, label=f"n={n}", color=color)

    plt.xlabel("alpha")
    plt.ylabel("Vn(alpha)")
    plt.title("Function Vn(alpha) for different n")
    plt.legend()
    plt.grid(True)
    #plt.show()

# Run the plot
plot_vn()
