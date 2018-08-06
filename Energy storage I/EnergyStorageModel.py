"""
Energy storage model class
Adapted from code by Donghun Lee (c) 2018

"""
from collections import namedtuple
import numpy as np
import pandas as pd

class EnergyStorageModel():
    """
    Base class for energy storage model
    """

    def __init__(self, state_variable, decision_variable, state_0, data, possible_decisions,
                 exog_info_fn=None, transition_fn=None, objective_fn=None):
        """
        Initializes the model

        :param state_variable: list(str) - state variable dimension names
        :param decision_variable: list(str) - decision variable dimension names
        :param state_0: dict - contains the information to populate initial state, including eta (the fraction of
               energy maintained when charging or discharging the battery) and battery capacity
        :param data: DataFrame - contains the price information for some time period at 1 hour increments
        :param possible_decisions: list - list of possible decisions we could make
        :param exog_info_fn: function - calculates relevant exogenous information
        :param transition_fn: function - takes in decision variables and exogenous information to describe how the state
               evolves
        :param objective_fn: function - calculates contribution at time t
        """

        self.initial_state = state_0
        self.state_variable = state_variable
        self.decision_variable = decision_variable
        self.data = data
        self.possible_decisions = possible_decisions
        self.State = namedtuple('State', state_variable)
        self.state = self.build_state(state_0)
        self.Decision = namedtuple('Decision', decision_variable)
        self.objective = 0.0
        self.states = [self.state]

    def build_state(self, info):
        """
        this function returns a state containing all the state information needed

        :param info: dict - contains all state information
        :return: namedtuple - a state object
        """
        return self.State(*[info[k] for k in self.state_variable])

    def build_decision(self, info, energy_amount):
        """
        this function returns a decision

        :param info: dict - contains all decision info
        :param energy_amount: float - amount of energy
        :return: namedtuple - a decision object

        """
        info_copy = {'buy': 0, 'hold': 0, 'sell': 0}
        # the amount of power that can be bought or sold is limited by constraints
        for k in self.decision_variable:
            if k == 'buy' and info[k] > (self.initial_state['battery_capacity'] -
                                         energy_amount) / self.initial_state['eta']:
                info_copy[k] = (self.initial_state['battery_capacity'] - energy_amount) / self.initial_state['eta']
            elif k == 'sell' and info[k] > energy_amount:
                info_copy[k] = energy_amount
            else:
                info_copy[k] = info[k]
        return self.Decision(*[info_copy[k] for k in self.decision_variable])

    def exog_info_fn(self, time):
        """
        this function simply returns the next price or the change in price (when we're reading in price information)

        :param time: int - time at which the state is at
        :return: float - price at the next time instance
        """
        # read in the next price from the dataset
        next_price = self.data.iat[time + 1, 4]
        # alternatively, we return the change in price, p_t - p_(t-1)
        # price_change = next_price - self.data.iat[time, 4]
        return next_price

    def transition_fn(self, time, decision):
        """
        this function takes in the decision and exogenous information to update the state

        :param time: int - time at which the state is at
        :param decision: namedtuple - contains all decision info
        :return: updated state
        """
        new_price = self.exog_info_fn(time)
        new_energy_amount = self.state.energy_amount + (self.initial_state['eta'] * decision.buy) - decision.sell
        state = self.build_state({'energy_amount': new_energy_amount,
                                  'price': new_price,
                                  'eta': self.initial_state['eta'],
                                  'battery_capacity': self.initial_state['battery_capacity']})
        return state

    def objective_fn(self, decision):
        """
        this function calculates the contribution, which depends on the decision and the price

        :param decision: namedtuple - contains all decision info
        :return: float - calculated contribution
        """
        obj_part = self.state.price * (decision.sell - decision.buy)
        return obj_part

    def step(self, time, decision):
        """
        this function steps the process forward by one time increment by updating the sum of the contributions
        and the state variable

        :param time: int - time at which the state is at
        :param decision: decision: namedtuple - contains all decision info
        :return: none
        """
        self.objective += self.objective_fn(decision)
        self.state = self.transition_fn(time, decision)
        self.states.append(self.state)

# unit testing
if __name__ == "__main__":
    # load energy price data from the Excel spreadsheet
    file = 'PJM_Historical_DA_RT_LMPs_05_to_11.xlsx'
    xl = pd.ExcelFile(file)
    raw_data = xl.parse('Raw Data')
    t = 0
    stop_time = 191

    # this is an example of creating a model, using a random policy, and running until the resource hits 0.
    # we put eta and battery capacity in the state variable to avoid 'floating' parameters
    state_variable = ['energy_amount', 'price', 'eta', 'battery_capacity']
    initial_state = {'energy_amount': 1,
                     'price': raw_data.iat[t, 4],
                     'eta': 0.9,
                     'battery_capacity': 20.0}
    decision_variable = ['buy', 'hold', 'sell']
    possible_decisions = [{'buy': 1, 'hold': 0, 'sell': 0}, {'buy': 0, 'hold': 0, 'sell': 1},
                          {'buy': 0, 'hold': 1, 'sell': 0}]
    M = EnergyStorageModel(state_variable, decision_variable, initial_state, raw_data, possible_decisions)

    # we run the process for the chosen time period - in this case, it is one week in one-hour increments
    while t != stop_time + 1:
        # implement a random decision policy
        if np.random.uniform() > 0.8:
            decision = M.possible_decisions[0]
        else:
            decision = M.possible_decisions[1]
        x = M.build_decision(decision, M.state.energy_amount)
        print("t={}, obj={}, state.energy_amount={}, state.price={}, x={}".format(t, M.objective, M.state.energy_amount,
                                                                                  M.state.price, x))
        M.step(t, x)

        # increment time
        t += 1