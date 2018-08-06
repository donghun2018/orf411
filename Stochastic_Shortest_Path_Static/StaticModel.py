"""
Stochastic Shortest Path Extension
Using point estimates

"""
from collections import namedtuple

import numpy as np


class StaticModel():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, s_0,
    exog_info, target_node, samplesize, alpha, policy=None, 
                 transition_fn=None, objective_fn=None, seed=20180529):
        """
        Initializes the model

        :param state_names: list(str) - state variable dimension names
        :param x_names: list(str) - decision variable dimension names
        :param s_0: dict - need to contain at least information to populate initial state using s_names
        :param exog_info_fn: function -
        :param transition_fn: function -
        :param objective_fn: function -
        :param seed: int - seed for random number generator
        """

        self.init_args = {seed: seed}
        self.prng = np.random.RandomState(seed)
        self.init_state = s_0
        self.state_names = state_names
        self.x_names = x_names
        self.State = namedtuple('State', state_names)
        self.state = self.build_state(s_0)
        self.Decision = namedtuple('Decision', x_names)
        
        # policy function, given by Bellman's equation
        self.policy = policy
        
        # value of objective function
        self.obj = 0.0
        
        # exogenous information = structure of the graph
        self.exog_info = exog_info
        
        # smoothing rate
        self.alpha = alpha
        
        # sample size for approximaions
        self.samplesize = samplesize
        
        # target node
        self.target_node = target_node

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])
    
    # returns an adjacent node that leads to the shortest path to the target_node
    def policy_fn(self):
        return self.policy[self.state[0]]
    
    def exog_info_fn(self, Node1, Node2):
        return self.exog_info[(Node1, Node2)]

    def transition_fn(self, decision):
        node = self.state[0]
        self.obj = self.obj + self.exog_info.distances[(node, decision[0])]
        self.state = self.build_state({'CurrentNode': decision[0], 'CurrentNodeLinks': self.exog_info.edges[decision[0]], 'TargetNode':self.target_node, 'SampleSize':self.samplesize, 'Alpha':self.alpha})
        return self.state

    def objective_fn(self):
        return self.obj


# Stochastic Graph class 
from collections import defaultdict
import math

class StochasticGraph:
    def __init__(self):
        self.nodes = list()
        self.edges = defaultdict(list)
        self.lower = {}
        self.distances = {}
        self.upper = {}

    def add_node(self, value):
        self.nodes.append(value)
    
    # create edge with normal weight w/ given mean and var
    def add_edge(self, from_node, to_node, lower, upper):
        self.edges[from_node].append(to_node)
        self.distances[(from_node, to_node)] = np.random.uniform(lower, upper)
        self.lower[(from_node, to_node)] = lower
        self.upper[(from_node, to_node)] = upper
    
    # return the expected length of the shortest paths w.r.t. given node
    def bellman(self, target_node):
        inflist = [math.inf]*len(self.nodes)
        # vt - value list at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
        # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])):
                        vt[v] = vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])
                        dt[v] = w 
        # print(vt)
        # print(g.distances)
        return(vt)   
    
    def truebellman(self, target_node):
        inflist = [math.inf]*len(self.nodes)
        # vt - value list at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
         # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + self.distances[(v, w)]):
                        vt[v] = vt[w] + self.distances[(v, w)]
                        dt[v] = w 
        # print(vt)
        # print(g.distances)
        return(vt)  
    # policy function based on point estimtes
    def pointestimate(self, target_node):
        inflist = [math.inf]*len(self.nodes)
        # vt - value list at time t for all the nodes w.r.t. to target_node
        vt = {k: v for k, v in zip(self.nodes, inflist)}
        vt[target_node] = 0
        
        # decision function for nodes w.r.t. to target_node
        dt = {k:v for k,v in zip(self.nodes, self.nodes)}
        
        # updating vt
        for t in range(1, len(self.nodes)):            
            for v in self.nodes:
                for w in self.edges[v]:
                    # Bellman' equation 
                    if (vt[v] > vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])):
                        vt[v] = vt[w] + 0.5*(self.lower[(v,w)] + self.upper[(v,w)])
                        dt[v] = w 
        # print(vt)
        # print(g.distances)
        return(dt)   
LO_UPPER_BOUND = 100
HI_UPPER_BOUND = 500
# number of nodes
n = 50
# probability of two nodes having edge
p = 0.2

def randomgraph(n, p):
    g = StochasticGraph()
    for i in range(n):
        g.add_node(str(i))
    for i in range(n):
        for j in range(n):
            q = np.random.uniform(0,1)
            if (i != j and q < p):
                lo = np.random.uniform(0, LO_UPPER_BOUND)
                hi = np.random.uniform(lo, HI_UPPER_BOUND)
                g.add_edge(str(i), str(j), hi, lo)
    return(g)


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

# create the model, where we use point estimates
M3 = StaticModel(state_names, decision_names, init_state, 
           exog_info, target_node, samplesize, alpha, g.pointestimate(target_node))

while M3.state[0] != M3.target_node:
    # calling policy and choosing a decision
    decision = M3.policy_fn()
    x = M3.build_decision({'NextNode': decision})
    # print current state
    #print("M3.state={}, obj={}, decision={}".format(M3.state, M3.obj, x))
    # transition to the next state w/ the given decision
    M3.transition_fn(x)
pass
print("M.state={}, obj={}, decision={}".format(M3.state, M3.obj, x))
TrueObj = M3.exog_info.truebellman(str(n-1))['0']
print('TrueObj = ', TrueObj)

