import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import pulp
from collections import Counter, defaultdict

from scipy.stats import poisson # for clique problem
from networkx.algorithms import bipartite # for clique problem
import random # for the demo
import gc # for memory management associated with the demo
import time # performance evaluation

# Shift assignment and housing assignment are essentially separate problems
# Each problem should have an associated utility for identifying what weakness needs to be satisfied
# when an optimal solution does not exist.

# Let's start with the duty assignment problem. Preferecnes will be determined via survey: let's assume
# we let people rank their preferences. Expect data to come in a CSV matrix whose rows are volunteers, 
# columns are duty shifts (day - hour/shift pairs) and values are preference ranks, where 1 is the most 
# desired shift, and empty cells denote blocked shifts for that volunteer.

def build_shifts_graph(df):
    """
    Takes as input a dataframe read in from a CSV, where the columns are given by labels like "day1|shft1"
    and the row labels give volunteer names.
    """
    g = nx.DiGraph()
    for day_shift in df.columns:
        for vol in df.index:
            val = df[day_shift][vol]
            if np.isnan(val):
                continue
            day, shift = day_shift.split('|') # Not sure exactly what to do with this. I think day should be a node attr
            g.add_edge(vol, day_shift, {'preference':int(val)})
    return g

def build_problem_from_graph(g):
    pass
    
if __name__ == '__main__':
    import pandas as pd
    
    fname = 'demo_schedule.csv'
    df = pd.DataFrame.from_csv(fname)
    