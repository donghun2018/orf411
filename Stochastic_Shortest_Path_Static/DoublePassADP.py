
# coding: utf-8

# In[ ]:


"""
Stochastic Shortest Path Extension
Using diuble pass ADP

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
p = 0.1

g = randomgraph(n, p)
while(g.bellman(str(n-1))['0'] == math.inf):
    g = randomgraph(n, p)

# create the model
start_node = '0'
target_node = str(n-1)
samplesize = 40
alpha = 0.05
state_names = ['CurrentNode', 'CurrentNodeLinks', 'TargetNode', 'SampleSize', 'Alpha']
init_state = {'CurrentNode': start_node, 'CurrentNodeLinks': g.edges[start_node],
                  'TargetNode': target_node, 'SampleSize': samplesize, 'Alpha':alpha}
decision_names = ['NextNode']
exog_info = g
    
M1 = StaticModel(state_names, decision_names, init_state,
               exog_info, target_node, samplesize, alpha)


# calculating approximation of values of post-decision states from n samples
# using double pass

# getting initial values by solving bellman's eqn for deterministic version
V_t = M1.exog_info.bellman(target_node)
# checking if target node can be reached from the starting node
IS_REACHABLE = True
if (V_t[M1.state[0]] == math.inf):
    IS_REACHABLE = False
    print('Target node cannot be reached from the starting node')
v = {}
l = 0
samplecount1 = {k:v for k,v in zip(M1.exog_info.distances.keys(),
                                   np.zeros(len(M1.exog_info.distances.keys())))}
sampledist1 = {k:v for k,v in zip(M1.exog_info.distances.keys(), 
                                  np.zeros(len(M1.exog_info.distances.keys())))}
while (l < M1.samplesize and IS_REACHABLE == True):
    # eqn 5.20
    x = {}
    for i in M1.exog_info.nodes:
        minvalue = math.inf
        argmin = None
        # iterate through all of the neighboring vertices of i
        for j in M1.exog_info.edges[i]:
            # sampling the distance between the two nodes
            smpldist = np.random.uniform(M1.exog_info.lower[(i,j)], 
                                         M1.exog_info.upper[(i,j)])
            sampledist1[(i, j)] = sampledist1[(i, j)] + smpldist
            samplecount1[(i, j)] = samplecount1[(i, j)] + 1
            smpldist = sampledist1[(i, j)] / samplecount1[(i, j)]

            val = smpldist + V_t[j]
            if val < minvalue:
                minvalue = val
                argmin = j
        x[i] = argmin

    # restoring the current shortest path
    path = []
    cur_node = start_node
    while(cur_node != target_node):
        path.append(cur_node)
        cur_node = x[cur_node]
    path.append(target_node)

    # traversing the path backwards
    rpath = list(reversed(path))
    v[target_node] = 0

    # eqn 5.21
    for i in range(1, len(path)):
        # sampling the distance between the two nodes in the path
        smpldist = np.random.uniform(M1.exog_info.lower[(rpath[i], rpath[i-1])], 
                                     M1.exog_info.upper[(rpath[i], rpath[i-1])])
        sampledist1[(rpath[i], rpath[i-1])] = sampledist1[(rpath[i], rpath[i-1])] + smpldist
        samplecount1[(rpath[i], rpath[i-1])] = samplecount1[(rpath[i], rpath[i-1])] + 1
        smpldist = sampledist1[(rpath[i], rpath[i-1])] / samplecount1[(rpath[i], rpath[i-1])]

        v[rpath[i]] = smpldist + v[rpath[i-1]]

    # smoothing the values
    # eqn 5.18
    for node in v.keys():
        V_t[node] = (1-alpha)*V_t[node] + v[node]*alpha

    l += 1


# eqn 5.15
# Bellman's w/ post decision states
val = {}
policy = {}
for node in M1.exog_info.nodes:
    minvalue = math.inf
    argmin = None
    for x in M1.exog_info.edges[node]:
        if (V_t[x] + M1.exog_info.distances[(node, x)] < minvalue):
            minvalue = V_t[x] + M1.exog_info.distances[(node, x)]
            argmin = x
    val[node] = minvalue
    policy[node] = argmin
M1.policy = policy

while M1.state[0] != M1.target_node:
    # calling policy and choosing a decision
    decision = M1.policy_fn()
    x = M1.build_decision({'NextNode': decision})
    # print current state
    print("M.state={}, obj={}, decision={}".format(M1.state, M1.obj, x))
    # transition to the next state w/ the given decision
    M1.transition_fn(x)
    # print("t={}, obj={}, s.resource={}".format(t, M1.obj, M1.state.resource))
pass
print("M.state={}, obj={}, decision={}".format(M1.state, M1.obj, x))
TrueObj = M1.exog_info.truebellman(str(n-1))['0']
print('TrueObj = ', TrueObj)

