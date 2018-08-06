"""
Two Newsvendor as a Learning Problem 

Author: Andrei Graur

This program implements a model for the two newsvendor 
problem where the field agent and the central command
both view the problem of choosing the right bias to add
or substract as a learning problem. Run the code with the
python command, no arguments given. 

"""

import numpy as np
import pandas as pd
import math
import xlrd

from TwoNewsvendor import Model_Field
from TwoNewsvendor import Model_Central

# the class implementing the objects that represent the 
# avaiable choices of the two agents, which are the biases to add
class Choice:
	def __init__(self, quantity, pen_estimate, precision_estimate, param_UCB):
		'''
		The function that initializes the choice object

		param: quantity - int: the quantity in units equal to the bias
		param: pen_estimate - float: the estimate of what the penalty will 
		be when we use this bias; we initialize it to 0 in main 
		param: precision_estimate - float: the estimate of what the 
		precision of next experiment of using this bias will be
		param: param_UCB - float: the tunable parameter that appears 
		in the notes as the "upper confidence bound" parameter at
		the Designing Policies setion from Learning for Diabetes Medication
		chapter.
		'''
		self.quantity = quantity
		self.estimate = pen_estimate
		self.exp_precision = precision_estimate
		self.exp_variance = 1 / float(precision_estimate)
		self.accumulated_precision = 0
		self.param_UCB = param_UCB
		self.perf_experiments = 0

	# the function that uploads the results of the experiment of trying
	# this bias and updates the corresponding beliefs about this choice
	def upload_results(self, pen_incurred):
		self.perf_experiments += 1
		n =  self.perf_experiments
		# update the variance
		if n > 2:
			new_stddev = (((n - 2.0) / float(n - 1)) * math.sqrt(self.exp_variance) +
						 (1.0 / n) * ((pen_incurred - self.estimate) ** 2))
			self.exp_variance = new_stddev ** 2
			self.exp_precision = 1 / float(self.exp_variance)
		# update estimate and experiment precision 
		self.estimate = ((self.estimate * self.accumulated_precision +
						pen_incurred * self.exp_precision) / 
						(self.accumulated_precision + self.exp_precision))
		self.accumulated_precision += self.exp_precision

	# the function that returns the bias attribute of this object
	def get_choice_quantity(self):
		return self.quantity

	# the cost function approxiamtion for this choice of bias
	def get_UCB_value(self, time):
		if self.perf_experiments == 0:
			UCB_val =  np.inf

		else:
			UCB_val = (self.estimate + self.param_UCB * 
					  math.sqrt(math.log(time) / self.perf_experiments))
		return UCB_val
	
	def get_nb_experiments(self):
		return self.perf_experiments

# the model for the field agent treating the problem as a 
# learning problem 
class Learning_model_field(Model_Field):
	def __init__(self, param_UCB, *args, **kwargs):
		self.choices = {}
		for value in range(11):
			self.choices[value] = Choice(value, 0, 1, param_UCB)
		super(Learning_model_field, self).__init__(*args, **kwargs)

	# the new transition function for the learning approach
	def transition_fn(self, state, pen_incurred, bias_applied, actual_demand, newEstimate):
		self.state = state.copy()
		# update the results of having tried out the used choice 
		choice_used = self.choices[bias_applied]
		choice_used.upload_results(pen_incurred)
		# update beliefs about the external source
		source_bias = self.state['estimate'] - actual_demand
		self.state['source_bias'] = self.state['source_bias'] * 0.9 + 0.1 * source_bias
		self.state['estimate'] = newEstimate
		return self.state

# the model for the central command treating the problem as a 
# learning problem
class Learning_model_central(Model_Central):
	def __init__(self, param_UCB, *args, **kwargs):
		self.choices = {}
		for value in range(-11, 1):
			self.choices[value] = Choice(value, 0, 1, param_UCB)
		super(Learning_model_central, self).__init__(*args, **kwargs)

	# the new transition function for the learning approach
	def transition_fn(self, state, pen_incurred, bias_applied, request, prev_field_bias):
		self.state = state.copy()
		self.state['field_request'] = request
		# update the results after having tried the used choice 
		choice_used = self.choices[bias_applied]
		choice_used.upload_results(pen_incurred)
		return self.state



