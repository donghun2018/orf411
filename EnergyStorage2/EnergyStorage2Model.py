"""
Energy Storage II Model

"""
from collections import namedtuple
import numpy as np
import math
from Forecast import Forecast

class EnergyStorage2Model():
	"""
	Base class for Energy Storage II model
	"""

	def __init__(self, state_names, x_names, forecast_names, forecasts, eta, s_0, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180529):
		"""
		Initializes the model
		
		:param state_variable: list(str) - state variable dimension names
		:param x_names: list(str) - decision variable dimension names
		:param forecast_names: list(str) - forecast variables names
		:param forecasts: dict - dictionary with all needed forecast objects
		:param eta: float - battery efficiency
		:param state_0: dict - needs to contain at least the information to populate initial state using state_names
		:param exog_info_fn: function - calculates relevant exogenous information
		:param transition_fn: function - takes in decision variables and exogenous information to describe how the state
			   evolves
		:param objective_fn: function - calculates contribution at time t
		:param seed: int - seed for random number generator
		"""

		self.init_args = {seed: seed}
		self.prng = np.random.RandomState(seed)
		self.init_state = s_0
		self.state_names = state_names
		self.x_names = x_names
		self.forecast_names = forecast_names
		self.State = namedtuple('State', state_names)
		self.state = self.build_state(s_0)
		self.Decision = namedtuple('Decision', x_names)
		self.forecasts = forecasts
		self.eta = eta
		self.obj = 0.0
		self.t = 0 # time counter (in months)

	# this function gives a state containing all the state information needed
	def build_state(self, info):
		return self.State(*[info[k] for k in self.state_names])

	# this function gives a decision 
	def build_decision(self, info):
		return self.Decision(*[info[k] for k in self.x_names])

	# this function gives the exogenous information that is dependent on a random process
	# computes the f_hat, chnage in the forecast over the horizon
	def exog_info_fn(self, decision):
		pass

	# this function takes in the decision and exogenous information to return
	# new state
	def transition_fn(self, decision, exog_info):
		new_state = {}
		new_state['R'] = self.state.R
		new_state['R'] += decision.x_wr
		new_state['R'] += decision.x_gr
		new_state['R'] -= decision.x_rg
		new_state['R'] -= decision.x_rl
		self.forecasts['f_L'].t_update()
		self.forecasts['f_G'].t_update()
		self.forecasts['f_W'].t_update()
		self.forecasts['f_P'].t_update()
		new_state['f_L'] = self.forecasts['f_L'].return_forecast()
		new_state['f_W'] = self.forecasts['f_W'].return_forecast()
		new_state['f_P'] = self.forecasts['f_P'].return_forecast()
		new_state['f_G'] = self.forecasts['f_G'].return_forecast()
		self.state = self.build_state(new_state)
	
	# this function calculates how much money we make
	def objective_fn(self, decision, exog_info):
		obj = (decision.x_wl + decision.x_gl + self.eta * decision.x_rl) * self.state.f_P[0] - (decision.x_gl + decision.x_gr - self.eta * decision.x_rg) * self.state.f_G[0]
		return obj
		
	# this method steps the process forward by one time increment by updating the sum of the contributions, the
	# exogenous information and the state variable
	def step(self, decision):
		exog_info = self.exog_info_fn(decision)
		
		# update objective
		self.obj += self.objective_fn(decision, exog_info)
		
		self.transition_fn(decision, exog_info)
		self.t_update()
		
	# Update method for time counter
	def t_update(self):
		self.t += 1
		return self.t
		
	# reset battery energy level
	def resetR(self):
		new_state = {}
		new_state['R'] = self.state.R
		new_state['f_L'] = self.forecasts['f_L'].return_forecast()
		new_state['f_W'] = self.forecasts['f_W'].return_forecast()
		new_state['f_P'] = self.forecasts['f_P'].return_forecast()
		new_state['f_G'] = self.forecasts['f_G'].return_forecast()
		self.state = self.build_state(new_state)