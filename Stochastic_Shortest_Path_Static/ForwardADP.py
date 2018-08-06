"""
Stochastic Shortest Path Extension
Using ADP with forward pass

"""
import numpy as np
import math
from StaticModel import (StaticModel, StochasticGraph, randomgraph)
from collections import (namedtuple, defaultdict)

LO_UPPER_BOUND = 100
HI_UPPER_BOUND = 500
# number of nodes
n = 50
# probability of two nodes having edge
p = 0.2

g = randomgraph(n, p)
while(g.bellman(str(n-1))['0'] == math.inf):
    g = randomgraph(n, p)


# create the mode
start_node = '0'
target_node = str(n-1)
samplesize = 40
alpha = 0.05
state_names = ['CurrentNode', 'CurrentNodeLinks', 'TargetNode', 'SampleSize', 'Alpha']
init_state = {'CurrentNode': start_node, 'CurrentNodeLinks': g.edges[start_node],
                  'TargetNode': target_node, 'SampleSize': samplesize, 'Alpha':alpha}
decision_names = ['NextNode']
exog_info = g
    
M2 = StaticModel(state_names, decision_names, init_state,
               exog_info, target_node, samplesize, alpha)


# calculating approximation of values of post-decision states from n samples
# using just forward pass

# getting initial values of post-decision states
# by solving bellman's eqn for deterministic version
V_t = M2.exog_info.bellman(target_node)
IS_REACHABLE = True
if (V_t[M2.state[0]] == math.inf):
    IS_REACHABLE = False
    print('Target node cannot be reached from the starting node')
v = {}
l = 0
samplecount2 = {k:v for k,v in zip(M2.exog_info.distances.keys(), 
                                   np.zeros(len(M2.exog_info.distances.keys())))}
sampledist2 = {k:v for k,v in zip(M2.exog_info.distances.keys(), 
                                  np.zeros(len(M2.exog_info.distances.keys())))}
while (l < M2.samplesize and IS_REACHABLE == True):
    for i in M2.exog_info.nodes:
        minvalue = math.inf
        argmin = None
        # iterate through all of the neighboring vertices of i
        for j in M2.exog_info.edges[i]:
            smpldist = np.random.uniform(M2.exog_info.lower[(i,j)], 
                                         M2.exog_info.upper[(i,j)])
            sampledist2[(i, j)] = sampledist2[(i, j)] + smpldist
            samplecount2[(i, j)] = samplecount2[(i, j)] + 1
            smpldist = sampledist2[(i, j)] / samplecount2[(i, j)]
            if (smpldist + V_t[j] < minvalue):
                minvalue = smpldist + V_t[j]
                argmin = j
        v[i] = minvalue
    # smooth the values
    # eqn 5.18
    v[M2.target_node] = 0
    for node in V_t.keys():
        V_t[node] = (1-alpha)*V_t[node] + v[node]*alpha
    l += 1


# In[ ]:


# eqn 5.15
# Bellman's w/ post decision states
val = {}
policy = {}
for node in M2.exog_info.nodes:
    minvalue = math.inf
    argmin = None
    for x in M2.exog_info.edges[node]:
        if (V_t[x] + M2.exog_info.distances[(node, x)] < minvalue):
            minvalue = V_t[x] + M2.exog_info.distances[(node, x)]
            argmin = x
    val[node] = minvalue
    policy[node] = argmin
M2.policy = policy


while M2.state[0] != M2.target_node:
    # calling policy and choosing a decision
    decision = M2.policy_fn()
    x = M2.build_decision({'NextNode': decision})
    # print current state
    print("M2.state={}, obj={}, decision={}".format(M2.state, M2.obj, x))
    # transition to the next state w/ the given decision
    M2.transition_fn(x)
    # print("t={}, obj={}, s.resource={}".format(t, M2.obj, M2.state.resource))
pass
print("M.state={}, obj={}, decision={}".format(M2.state, M2.obj, x))
TrueObj = M2.exog_info.truebellman(str(n-1))['0']
print('TrueObj = ', TrueObj)

