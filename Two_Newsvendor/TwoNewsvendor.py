"""
Author: Andrei Graur

This program implements the basic model for the two newsvendor problem. 
This code does not belong to the driverscript 


"""
from collections import namedtuple

import numpy as np
import pandas as pd
import math
import xlrd

class Model_Field():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, s_0, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180529):
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
        self.exog_info = {}
        self.o_q = s_0['o_q']
        self.u_q = s_0['u_q']
        self.total_variance_central = 0.0
        self.lambda_central = 1.0
        self.total_variance_source = 0.0
        self.lambda_source = 1.0

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])

    def exog_info_fn(self, decision_central, demand):
        exog_info = []
        exog_info.append(decision_central) 
        exog_info.append(demand) 
        return exog_info

    def transition_fn(self, state, central_bias, actual_demand, newEstimate):
        self.state = state.copy()
        # update beliefs about central agent
        self.state['central_bias'] = self.state['central_bias'] * 0.9 + 0.1 * central_bias
        self.total_variance_central = 0.9 * self.total_variance_central + 0.1 * ((central_bias - self.state['central_bias']) ** 2)
        self.lambda_central = self.lambda_central * 0.81 + 0.01
        self.state['central_var'] = (self.total_variance_central - (self.state['central_bias'] ** 2)) / (1 + self.lambda_central)
        if self.state['central_var'] < 0:
            self.state['central_var'] = 0
                                 
        # update beliefs about the external source
        source_bias = self.state['estimate'] - actual_demand
        self.state['source_bias'] = self.state['source_bias'] * 0.9 + 0.1 * source_bias
        self.total_variance_source = 0.9 * self.total_variance_source + 0.1 * ((source_bias - self.state['source_bias']) ** 2)
        self.lambda_source = self.lambda_source * 0.81 + 0.01
        self.state['source_var'] = (self.total_variance_source - (self.state['source_bias'] ** 2)) / (1 + self.lambda_source)
        if self.state['source_var'] < 0:
            self.state['source_var'] = 0

        self.state['estimate'] = newEstimate
        return self.state

    def objective_fn(self, decision, exog_info):
        allocated = exog_info['allocated_quantity']
        demand = exog_info['demand']
        penalty = 0
        if allocated > demand:
            penalty = -(allocated - demand) * self.o_q
        else :
            penalty = (allocated - demand) * self.u_q
        return penalty 


    def showCentralBias(self):
        return self.state['central_bias']

    def showSourceBias(self):
        return self.state['source_bias']


class Model_Central():
    """
    Base class for model
    """

    def __init__(self, state_names, x_names, s_0, exog_info_fn=None, transition_fn=None, objective_fn=None, seed=20180529):
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
        self.o_qPrime = s_0['o_qPrime']
        self.u_qPrime = s_0['u_qPrime']
        self._lambda_ = 0.01
        self.total_variance = 0

    def build_state(self, info):
        return self.State(*[info[k] for k in self.state_names])

    def build_decision(self, info):
        return self.Decision(*[info[k] for k in self.x_names])

    def exog_info_fn(self, req_quantity, demand):
        return demand

    def transition_fn(self, state, request, prev_field_bias):
        self.state = state.copy()
        self.state['field_request'] = request
        # update the beliefs of the central agent 
        self.total_variance = 0.9 * self.total_variance + 0.1 * ((prev_field_bias - self.state['field_bias']) ** 2)
        self.state['field_bias'] = self.state['field_bias'] * 0.9 + 0.1 * prev_field_bias
        self.state['field_var'] = (self.total_variance - (self.state['field_bias'] ** 2)) / (1 + self._lambda_)
        self._lambda_ = self._lambda_ * 0.81 + 0.01
        if self.state['field_var'] < 0:
            self.state['field_var'] = 0

        return self.state

    def objective_fn(self, decision, exog_info):
        q_supplied = decision 
        if q_supplied > exog_info:
            penalty = -(q_supplied - exog_info) * self.o_qPrime
        else:
            penalty = (q_supplied - exog_info) * self.u_qPrime
        return penalty

    def showBias(self):
        return self.state['field_bias']

