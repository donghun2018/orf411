"""
Stochastic Shortest Paths - Dynamic
Static Model

The code implementing the basic model for the Static 
Version. Run it with the python command in terminal. 

Author: Andrei Graur 

"""
from collections import namedtuple

import math
import numpy as np
import pandas as pd
import xlrd


class StaticModel():
	"""
	Base class for the static model
	"""

	def __init__(self, state_names, x_names, s_0, Horizon, vertexCount, seed=20180529):
		"""
		Initializes the model

		:param state_names: list(str) - state variable dimension names
		:param x_names: list(str) - decision variable dimension names
		:param s_0: dict - contains the inital state information
		:param s_0[meanCosts]: dict- meanCosts[k][l] is the mean of the cost on the link k-l 
		:param s_0[spreads]: dict - spreads[k][l] represents the spread of the distribution of  
		cost on link k-l 
		:param Horizon: int - the horizon over which we are looking ahead
		:param vertexCount - the number of nodes in our network 
		:param seed: int - seed for random number generator
		"""

		self.init_args = {seed: seed}
		self.prng = np.random.RandomState(seed)
		self.init_state = s_0
		self.state_names = state_names
		self.x_names = x_names
		self.State = namedtuple('State', state_names)
		self.state = self.build_state(s_0)
		self.meanCosts = s_0['meanCosts']
		self.spreads = s_0['spreads']
		self.Decision = namedtuple('Decision', x_names)
		self.horizon = Horizon
		self.vertexCount = vertexCount
		self.decision = [None] * vertexCount

	def build_state(self, info):
		return self.State(*[info[k] for k in self.state_names])

	def build_decision(self, node, decision):
		self.decision[node] = decision
		return self.decision

	# exog_info_fn: function - returns the real experienced cost of traversing a link 
	# from 'fromNode' to 'toNode' 
	def exog_info_fn(self, arrivalTime, fromNode, toNode):
		spread = self.spreads[fromNode][toNode]
		deviation = np.random.uniform(- spread, spread) * self.meanCosts[fromNode][toNode]
		cost = self.meanCosts[fromNode][toNode] + deviation
		return cost

	# transition_fn: function - updates the state within the model and returns new state
	def transition_fn(self, state, decision):
		self.state = state
		node = self.state['node']
		self.state['node'] = decision
		return self.state

	# :param objective_fn: function - returns the cost we would experience by taking 'decision'
	# as our next node from the current state 'state'
	def objective_fn(self, decision, state, time):
		node = state['node']
		cost = self.exog_info_fn(time, node, decision)
		return cost 


# the function that updates the cost estimates when we 
# move from time t to time t + 1
def t_estimated_costs(neighbors, meanCosts, spreads, vertexCount):
	estimated_costs = {}
	for k in range(vertexCount):
		costsFromK = {}
		for l in neighbors[k]:
			spread = spreads[k][l]
			deviation = np.random.uniform(- spread, spread) * meanCosts[k][l]
			costsFromK[l] = meanCosts[k][l] + deviation
		estimated_costs[k] = costsFromK
	estimated_costs[r][r] = 0
	return estimated_costs



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
	'''
	initialize the lists of vertices, their neighbors and the mean
	costs of the links from each vertex, as well as the spreads of
	each link. The spread determines the length of a tail for the 
	probability distribution of a certain cost. Each cost is modeled as
	a random variable with unifrom distribution in the interval 
	[(1 - spread) * mean, (1 + spread) * mean]
	'''
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

		
	# We need to add the dummy link of cost 0
	neighbors[r].append(r)
	meanCosts[r][r] = 0

	state_names = ['node']
	init_state = {'node': q, 'meanCosts': meanCosts, 'spreads': spreads}
	decision_names = ['nextNode']

	M = StaticModel(state_names, decision_names, init_state, Horizon, vertexCount)

	# the matrix with decisions to be made for each node
	decisions = [ ([0] * vertexCount) for row in range(Horizon + 1) ]


	node = q
	print("We are at node %d" % node)
	time = 0
	actualCost = 0
	state = {'node': q}
	while state['node'] != r:
		node = state['node']

		# initialize the value costs at different nodes at different times to infinity
		V = np.ones((Horizon + 1, vertexCount)) * np.inf
		# make the costs at the destination 0
		for t_prime in range(Horizon + 1):
			V[t_prime][r] = 0

		costArray = t_estimated_costs(neighbors, meanCosts, spreads, vertexCount)
		# the algortihm that uses the "stepping backwards in time" method
		lookAheadTime = Horizon - 1
		while lookAheadTime >= time:
			for k in vertices:
				# find the solutions to Bellman's eq. that are shown 
				# in 5.22 and 5.23
				argMin = - 1
				minVal = np.inf
				for l in neighbors[k]:
					if minVal >= V[lookAheadTime + 1][l] + costArray[k][l]:        
						argMin = l
						minVal = V[lookAheadTime + 1][l] + costArray[k][l]
				# updating the solutions to the equations
				V[lookAheadTime][k] = minVal
				decisions[lookAheadTime][k] = argMin
			lookAheadTime -= 1

		decision = decisions[time][node]
		M.build_decision(node, decision)
		print("We are at node %d" % decision)
		print("Cost of the last link was %f" % M.objective_fn(decision, state, time))
		actualCost += M.objective_fn(decision, state, time)
		time += 1
		state = M.transition_fn(state, decision)

	print("Cost of shortest path was %f" % actualCost)

	pass
