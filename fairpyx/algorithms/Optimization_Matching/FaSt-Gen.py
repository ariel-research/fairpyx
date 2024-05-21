"""
    "OnAchieving Fairness and Stability in Many-to-One Matchings", by Shivika Narang, Arpita Biswas, and Y Narahari (2022)

    Programmer: Hadar Bitan, Yuval Ben-Simhon
"""

from fairpyx import Instance, AllocationBuilder, ExplanationLogger
import logging
logger = logging.getLogger(__name__)

def FaStGen(I:tuple)->dict:
    """
    Algorithem 3-FaSt-Gen: finding a match for the general ranked valuations setting.
    

    :param I: A presentation of the problem, aka a tuple that contain the list of students(S), the list of colleges(C) when the capacity
    of each college is n-1 where n is the number of students, student valuation function(U), college valuation function(V).

    >>> from fairpyx.adaptors import divide
    >>> S = {"s1", "s2", "s3", "s4", "s5", "s6", "s7"}
    >>> C = {"c1", "c2", "c3"}
    >>> V = {
        "c1" : ["s1","s2","s3","s4","s5","s6","s7"], 
        "c2" : ["s2","s4","s1","s3","s5","s6","s7"], 
        "c3" : [s3,s5,s6,s1,s2,s4,s7]}           #the colleges valuations                    
    >>> U = {
        "s1" : ["c1","c3","c2"], 
        "s2" : ["c2","c1","c3"], 
        "s3" : ["c1","c3","c2"], 
        "s4" : ["c3","c2","c1"],
        "s5" : ["c2","c3","c1"], 
        "s6" :["c3","c1","c2"], 
        "s7" : ["c1","c2","c3"]}        #the students valuations                 
    >>> I = (S, C, U ,V)
    >>> FaStGen(I)
    {'c1' : ['s1','s2'], 'c2' : ['s3','s4','s5']. 'c3' : ['s6','s7']}
    """


if __name__ == "__main__":
    import doctest, sys
    print(doctest.testmod())
