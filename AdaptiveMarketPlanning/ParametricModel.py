"""
Adaptive Market Planning Model for variable price subclass

"""
		
from collections import namedtuple
from AdaptiveMarketPlanningModel import AdaptiveMarketPlanningModel

import numpy as np

class ParametricModel(AdaptiveMarketPlanningModel):
	"""
	Subclass for Adaptive Market Planning
	"""

	def __init__(self, state_names, x_names, s_0, cost = 1.0, price_low = 1.0, price_high = 10.0, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180613):
		"""
		Initializes the model

		See Adaptive Market Planning Model for more details
        """
		super().__init__(state_names, x_names, s_0, cost = cost, exog_info_fn=exog_info_fn, transition_fn=transition_fn, objective_fn=objective_fn, seed=seed)
		self.past_derivative = np.array([0, 0, 0])
		self.low = price_low
		self.high = price_high
	
	# returns order quantity for a given price and theta vector
	def order_quantity_fn(self, price, theta):
		return theta[0] + theta[1] * price + theta[2] * price ** (-2)
	
	# returns derivative for a given price and theta vector
	def derivative_fn(self, price, theta):
		return np.array([1, price, price ** (-2)])

	# this function takes in the decision and exogenous information to return
	# new state
	def transition_fn(self, decision, exog_info):
	
		# compute derivative and update theta
		derivative = np.array([0, 0, 0])
		if self.order_quantity_fn(self.state.price, self.state.theta) < exog_info['demand']:
			derivative = (self.state.price - self.cost) * self.derivative_fn(self.state.price, self.state.theta)
		else:
			derivative = (- self.cost) * self.derivative_fn(self.state.price, self.state.theta)
		
		new_theta = self.state.theta + decision.step_size * derivative
	
		new_counter = self.state.counter + 1 if np.dot(self.past_derivative, derivative) < 0 else self.state.counter
		self.past_derivative = derivative
		
		# generate random price
		new_price = self.prng.uniform(self.low, self.high)
		
		return {"counter": new_counter, "price": new_price, "theta": new_theta}

	# this function calculates how much money we make
	def objective_fn(self, decision, exog_info, transition_info):
		obj_part = self.state.price * min(self.order_quantity_fn(self.state.price, self.state.theta), exog_info['demand']) - self.cost * self.order_quantity_fn(self.state.price, self.state.theta)
		return obj_part