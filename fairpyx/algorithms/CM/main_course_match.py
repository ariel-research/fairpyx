"""
Course Match: A Large-Scale Implementation ofApproximate Competitive Equilibrium fromEqual Incomes for Combinatorial Allocation,
by Eric Budish,a Gérard P. Cachon,b Judd B. Kessler,b Abraham Othmanb
June 2, 2016
https://pubsonline.informs.org/doi/epdf/10.1287/opre.2016.1544

Programmer: Naama Shiponi and Ben Dabush
Date: 1/6/2024
"""
from fairpyx.instances import Instance
from fairpyx.allocations import AllocationBuilder

def course_match_algorithm(instance: Instance):
    """
    Perform the Course Match algorithm to find the best course allocations.
    
    :param instance: (Instance)

    :return: (dict) course allocations
    """
    pass



if __name__ == "__main__":
    import doctest
    doctest.testmod()