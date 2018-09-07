"""
Adaptive Market Planning Driver Script

"""
		
from collections import namedtuple
from AdaptiveMarketPlanningModel import AdaptiveMarketPlanningModel
from AdaptiveMarketPlanningPolicy import AdaptiveMarketPlanningPolicy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# this is an example of creating a model and running a simulation for a certain trial size

	# define state variables
	state_names = ['order_quantity', 'counter']
	init_state = {'order_quantity': 0, 'counter': 0}
	decision_names = ['step_size']
	
	# read in variables from excel file
	file = 'Base parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('parameters')
	cost = data.iat[0, 2]
	trial_size = np.rint(data.iat[1, 2]).astype(int)
	price = data.iat[2, 2]
	theta_step = data.iat[3, 2]
	
	# initialize model and store ordered quantities in an array
	M = AdaptiveMarketPlanningModel(state_names, decision_names, init_state, price, cost)
	order_quantity = [init_state['order_quantity']]
	
	# use Kesten's rule to make decision
	for i in range(trial_size):
		M.step(AdaptiveMarketPlanningPolicy(M, theta_step).kesten_rule())
		order_quantity.append(M.state.order_quantity)

	# plot results
	plt.xlabel("time (log scale)")
	plt.ylabel("order quantity")
	plt.title("Kesten's rule")
	time = np.arange(1, trial_size + 2)
	plt.plot(np.log10(time), time * 0 + np.log(price) * 100, label = "Analytical solution")
	plt.plot(np.log10(time), order_quantity, label = "Kesten's Rule")
	plt.legend()
	plt.show()