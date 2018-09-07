"""
Parametric Model Driver Script

"""
	
from collections import namedtuple
from ParametricModel import ParametricModel
from AdaptiveMarketPlanningPolicy import AdaptiveMarketPlanningPolicy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
	# this is an example of creating a model and running a simulation for a certain trial size
	
	# define state variables
	state_names = ['counter', 'price', 'theta']
	init_state = {'counter': 0, 'price': 5, 'theta': np.array([1, 1, 1])}
	decision_names = ['step_size']
	
	# read in variables from excel file
	file = 'ParametricModel parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('parameters')
	cost = data.iat[0, 2]
	trial_size = np.rint(data.iat[1, 2]).astype(int)
	price_low = data.iat[2, 2]
	price_high = data.iat[3, 2]
	theta_step = data.iat[4, 2]

	# initialize model and run simulations
	M = ParametricModel(state_names, decision_names, init_state, cost, price_low = price_low, price_high = price_high)
		
	for i in range(trial_size):
		M.step(AdaptiveMarketPlanningPolicy(M, theta_step).kesten_rule())
	
	# plot results
	price = np.arange(price_low, price_high, 0.1)
	optimal = np.log(price) * 100
	plt.plot(price, optimal, color = 'green', label = "analytical solution")
	order_quantity = [M.order_quantity_fn(k, M.state.theta) for k in price]
	plt.plot(price, order_quantity, color = 'blue', label = "parametrized solution")
	plt.legend()
	plt.show()