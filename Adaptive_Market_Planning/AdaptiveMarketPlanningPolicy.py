"""
Adaptive Market Planning Policy class

"""
		
from collections import namedtuple

import numpy as np
from AdaptiveMarketPlanningModel import AdaptiveMarketPlanningModel

class AdaptiveMarketPlanningPolicy():
	"""
	Base class for policy
	"""

	def __init__(self, AdaptiveMarketPlanningModel, theta_step):
		"""
		Initializes the model

		:param AdaptiveMarketPlanningModel: AdaptiveMarketPlanningModel - model to construct decision for
		:param theta_step: float - theta step variable
        """
		
		self.M = AdaptiveMarketPlanningModel
		self.theta_step = theta_step

	# returns decision based on harmonic step size policy
	def harmonic_rule(self):
		return self.M.build_decision({'step_size': self.theta_step / (self.theta_step + self.M.t - 1)})
		
	# returns decision based on Kesten's rule policy
	def kesten_rule(self):
		return self.M.build_decision({'step_size': self.theta_step / (self.theta_step + self.M.state.counter - 1)})
