"""
Stochastic Shortest Paths - Dynamic
Dynamic Model - search for the parameter theta, 
which represents the percentile of the distribution 
of each cost to use to make sure we get a penalty as
small as possible. Run it just by using the python command.

Author: Andrei Graur 

"""
from collections import namedtuple
from DynamicModel import t_estimated_costs
from DynamicModel import DynamicModel

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd



    
'''
the function that takes as arguments the percentile we are going to
use, theta (espressed as a value in [0,1]), the spread for a link and
the mean cost of that link and returns the value corresponding to the 
theta precentile of the interval [(1 - spread) * mean, (1 + spread) * mean]
'''
def use_percentile_val(theta, spread, mean):
    point_val = 1 - spread + (2 * spread) * theta
    used_cost = mean * point_val
    return used_cost


'''
the function for running trials; it simulates solving the problem a bunch of 
times (nbTrials times), then takes the squared mean of the costs incurred, 
and then returns that mean value
'''
def runTrials(nbTrials, theta, neighbors, time_dep_means, spreads, vertexCount):
    deadline = 80
    sum_of_squares = 0.0
    # the matrix with decisions to be made for each node and for each time
    decisions = [ ([0] * vertexCount) for row in range(Horizon + 1) ]
    for i in range(nbTrials):
        node = q
        # print("We are at node %d" % node)
        time = 0
        totalCost = 0
        state = {'node': q}
        while state['node'] != r:
            node = state['node']

            # initialize the value costs at different nodes at different times to infinity
            V = np.ones((Horizon + 1, vertexCount)) * np.inf
            # make the costs at the destination 0
            for t_prime in range(Horizon + 1):
                V[t_prime][r] = 0

            costArray = t_estimated_costs(neighbors, time_dep_means, spreads, vertexCount)
            # the algortihm that uses the "stepping backwards in time" method
            lookAheadTime = Horizon - 1
            while lookAheadTime >= time:
                for k in range(vertexCount):
                    # find the solutions to Bellman's eq. that are shown 
                    # in 5.22 and 5.23
                    argMin = - 1
                    minVal = np.inf
                    for l in neighbors[k]:
                        spread = spreads[k][l]
                        mean = costArray[lookAheadTime][k][l]
                        if minVal >= V[lookAheadTime + 1][l] + use_percentile_val(theta, spread, mean):        
                            argMin = l
                            minVal = V[lookAheadTime + 1][l] + use_percentile_val(theta, spread, mean)
                    # updating the solutions to the equations
                    V[lookAheadTime][k] = minVal
                    decisions[lookAheadTime][k] = argMin
                lookAheadTime -= 1
            decision = decisions[time][node]
            M.build_decision(node, decision)
            totalCost += M.objective_fn(decision, state, time)
            time += 1
            state = M.transition_fn(state, decision)

        latenessSquared = 0
        if totalCost > deadline:
            latenessSquared = (totalCost - deadline) ** 2
        sum_of_squares += latenessSquared
    penalty = math.sqrt(sum_of_squares / nbTrials)
    return penalty


if __name__ == "__main__":
    
    # First, we read the network information
    raw_data = pd.read_excel("Network_Information3.xlsx", sheet_name="Network")

    meanCosts = {}
    spreads = {}
    neighbors = {}
    vertices = []
    vertexCount = math.floor(raw_data['Graph_size'][0])

    Horizon = vertexCount + 1
    # these are the indices of the start and end nodes
    q = 0
    r = vertexCount - 1

    for i in range(vertexCount):
        vertices.append(i)
        neighbors[i] = []
        meanCosts[i] = {}
        spreads[i] = {}

    for i in raw_data.index: 
        fromNode = raw_data['From'][i]
        toNode = raw_data['To'][i]
        neighbors[fromNode].append(toNode)
        meanCosts[fromNode][toNode] = raw_data['Cost'][i]
        spreads[fromNode][toNode] = raw_data['Spreads'][i]
    spreads[r][r] = 0


    # read the time dependent mean costs
    time_dep_means = {}
    for t in range(Horizon + 1):
    	list_at_t = {}
    	for k in range(vertexCount):
    		list_at_t[k] = {}
    	for i in raw_data.index: 
    		fromNode = raw_data['From'][i]
    		toNode = raw_data['To'][i]
    		column_name = "Time " + str(t) + " costs"
    		list_at_t[fromNode][toNode] = raw_data[column_name][i]
    	time_dep_means[t] = list_at_t 
    for t in range(Horizon + 1):
        time_dep_means[t][r][r] = 0
        
    # We need to add the dummy link of cost 0
    neighbors[r].append(r)
    meanCosts[r][r] = 0

    state_names = ['node']
    init_state = {'node': q, 'time_dep_means': time_dep_means, 'spreads': spreads}
    decision_names = ['nextNode']

    M = DynamicModel(state_names, decision_names, init_state, Horizon, vertexCount)

    x = []
    y = []
    data_points = 11
    for n in range(data_points):
        # randomly generate the actual costs for the links estimated for different time frames
        # with percentile theta
        theta = n * 1.0 / (data_points - 1)
        x.append(theta)
        penalty = runTrials(5000, theta, neighbors, time_dep_means, spreads, vertexCount)
        y.append(penalty)

        print("Total penalty with parameter {0} is {1}".format(theta, penalty))
    plt.title("Search for best percentile for dynamic model")
    plt.xlabel("Percentile")
    plt.ylabel("Penalty")
    plt.plot(x, y)
    plt.show()

    pass
