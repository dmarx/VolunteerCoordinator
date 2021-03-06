import numpy as np
import networkx as nx
import pulp
from collections import defaultdict

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
            if vol == 'shift_needs':
                continue
            val = df[day_shift][vol]
            if np.isnan(val):
                continue
            #day, shift = day_shift.split('|') # Not sure exactly what to do with this. I think day should be a node attr
            g.add_edge(vol, day_shift, {'cost':int(val)})
    d = dict(df.T['shift_needs'])
    nx.set_node_attributes(g, 'supply', d)
    vols = list(df.index)
    vols.remove('shift_needs')
    nx.set_node_attributes(g, 'supply', {v:-1 for v in vols})
    #g = add_dummy_supply_sink(g, ['supply'])
    return g
    

def add_dummy_supply_sink(g, attr_name, dummy_name='DummySinkNode'):
    """
    Given a problem where demand != supply, we need to balance supply and demand by adding a node connected to all
    supply or demand nodes to balance the difference. An edge directed towards this node essentially represents a recommendation 
    for storage at the supply node or unsatisified demand at the demand node. For this problem, storage cost will be zero
    but it does not necessarily need to be.
    
    Whether or not a dummy node is added, the input graph is always returned as a directed graph.
    """
    g0 = g.to_directed()
    g = g.to_directed()
    g.add_node(dummy_name)
    balanced = True
    for a in attr_name:
        supply = 0
        demand = 0
        for n,v in nx.get_node_attributes(g, a).items():
            if v>0:
                g.add_edge(n, dummy_name, {'cost':0})
                supply += v
            elif v<0:
                g.add_edge(dummy_name, n, {'cost':0})
                demand += v
        delta = supply + demand # demand is negative valued
        if delta != 0:
            balanced=False
        nx.set_node_attributes(g, a, {dummy_name:-delta})
    if balanced:
        return g0
    return g

    
def lp_assignment_from_pd(df, attr_name=['supply'], dummy_name='DummySinkNode'): 
    """
    Workhorse function for LP solution of transport problems represented in a graph. Currently only supports a single
    node attribute, but will expand function in future to operate on a list of node attributes.
    """    
    g = build_shifts_graph(df)
    g = add_dummy_supply_sink(g, attr_name)
    
    in_paths = defaultdict(list)
    out_paths = defaultdict(list)
    for p,q in g.edges_iter():
        out_paths[p].append((p,q))
        in_paths[q].append((p,q))
    
    vols = list(df.index)
    vols.remove('shift_needs')
    shifts = df.columns
    
    prob = pulp.LpProblem("Supply Chain", pulp.LpMinimize)
    
    x = pulp.LpVariable.dicts("takeShift", (vols, shifts), 
                            lowBound = 0 ,
                            upBound = 1,
                            cat = pulp.LpInteger)
    
    # Objective: minimize the sum of the costs of all accepted shifts.
    prob += sum(x[v][s]*g[v][s]['cost'] for v,s in g.edges_iter() if dummy_name not in v and dummy_name not in s), "objective"
    
    # Add constraint that net supply is satisfied across the graph.
    for shift in shifts:
        demand = nx.get_node_attributes(g, 'supply')[shift]
        prob += demand - sum([x[v][s] for (v,s) in in_paths[shift]]) == 0, "satisfiedShiftConstr_{}".format(shift)
    
    prob.solve()
    
    return prob
    
def extract_schedule(solution):
    results = [(str(v).split('_'), v.varValue) for v in prob.variables() if v.varValue==1]
    [a.append(v) for a,v in results]
    dt = pd.DataFrame([a for a,v in results])
    dt.columns = ['action','volunteer','shift','value']
    dt = dt.sort_values(by='shift')
    dt['x'] = 'x'
    return {'for_volunteers':dt.pivot(index='volunteer', columns='shift', values='x'), 
            'for_organizers':dt[['shift','volunteer']]}
    # This output format is convenient for the volunteers: they can check their name on the side
    # and quickly find their shift. I should really do a version that is convenient for me or
    # whoever is running things. Right now, just sorting on the 'shift' column and returning dt
    # basically does it, but I feel like there's a more appealing way I could do it. I should
    # also merge in contact information or something.
    #
    # Additionally, I should have a method that takes an existing solution as input and let's me specify
    # shifts that won't be satisfied so I can find alternates (by fixing every other position in the graph
    # (or at least the ones that have already been satisfied) and finding a new solution that doesn't impose
    # too much on everyone else). This may require a new objective function: instead of (or in addition to) 
    # minimizing the overall inconvenience, I could try to minimize the number of people I inconvenience 
    # with respect to the original optimal solution. 
    #
    # Also, I still need a utility for identifying unsatisfied demand so I can figure out what shifts I still 
    # need to fill while I'm finding volunteers. This probably just entails running the solver and determining
    # which shifts are connected to the dummy sink. Would be useful to know the cost-by-day as well, so I could
    # see which days seem to be unpopular and try to find people to work those days for whom it is more convenient
    # than the people in the current solution.
    #
    # It would probably be a good idea to maintain some kind of database component as well so I can associated
    # contact methods (emails, phone numbers) with volunteers. Maybe even automate asking them what shifts work
    # and so on. Shit, I could turn this into a whole website.
    #
    # I should move these ideas to github. If I end up turning this into a website, I should probably take this off 
    # github. Maybe move it to bitbucket where it'll be private.
    
if __name__ == '__main__':
    import pandas as pd
    
    fname = 'demo_schedule.csv'
    df = pd.DataFrame.from_csv(fname)
    solution = lp_assignment_from_pd(df)
    sched = extract_schedule(solution)
    sched['for_volunteers'].to_csv('demo_solution-volunteers.csv')
    sched['for_organizers'].to_csv('demo_solution-organizers.csv')
    