"""
Backward dynamic programming class
"""
from EnergyStorageModel import EnergyStorageModel as ESM
import numpy as np
import pandas as pd
from bisect import bisect
import matplotlib.pyplot as plt
import math
import time

class BDP():
    """
    Base class to implement backward dynamic programming
    """

    def __init__(self, discrete_prices, discrete_energy, price_changes, discrete_price_changes,
                 f_p, stop_time, model):
        """
        Initializes the model

        :param discrete_prices: list - list of discretized prices
        :param discrete_energy: list - list of discretized energy amounts
        :param price_changes: list - list of price changes
        :param discrete_price_changes: list - list of discretized price changes
        :param f_p: ndarray - contains f(p) values
        :param stop_time: int - time at which loop terminates
        :param model: energy storage model

        """
        self.discrete_energy = discrete_energy
        self.discrete_prices = discrete_prices
        self.price_changes = price_changes
        self.discrete_price_changes = discrete_price_changes
        self.f_p = f_p
        self.time = stop_time - 1
        self.model = model
        self.terminal_contribution = 0

    def contribution(self, state, decision):
        """
        this function computes the contribution based on a state and a decision

        :param state: namedtuple - the state of the model at a given time
        :param decision: namedtuple - contains all decision info
        :return: float - calculated contribution
        """
        contribution = state.price * (decision.sell - decision.buy)
        return contribution

    def state_transition(self, state, decision, exog_info):
        """
        this function tells us what state we transition to if we are in some state and make a decision
        (restricted to states in possible_states)

        :param state: namedtuple - the state of the model at a given time
        :param decision: namedtuple - contains all decision info
        :param exog_info: any exogenous info
        :return: new state object
        """
        new_price = state.price + exog_info
        if new_price <= min(self.discrete_prices):
            adjusted_new_price = min(self.discrete_prices)
        elif new_price >= max(self.discrete_prices):
            adjusted_new_price = max(self.discrete_prices)
        else:
            index = bisect(self.discrete_prices, new_price)
            adjusted_new_price = self.discrete_prices[index]

        new_energy = state.energy_amount + (self.model.initial_state['eta'] * decision.buy) - decision.sell
        adjusted_new_energy = math.ceil(new_energy)

        if len(state) == 4:
            new_state = self.model.build_state({'energy_amount': adjusted_new_energy,
                                                'price': adjusted_new_price,
                                                'eta': self.model.initial_state['eta'],
                                                'battery_capacity': self.model.initial_state['battery_capacity']
                                                })
        elif len(state) == 5:
            prev_price = state.price
            new_state = self.model.build_state({'energy_amount': adjusted_new_energy,
                                            'price': adjusted_new_price,
                                            'prev_price': prev_price,
                                            'eta': self.model.initial_state['eta'],
                                            'battery_capacity': self.model.initial_state['battery_capacity']})
        #print("new_price={}, adjusted_new_price={}, new_energy={}, adjusted_new_energy={}".format(new_price,
                                                                                                  #adjusted_new_price,
                                                                                                  #new_energy,
                                                                                                  #adjusted_new_energy))
        return new_state

    def bellman_2D(self):
        """
        this function computes the value function using Bellman's equation for a 2D state variable

        :return: list - list of contribution values
        """

        # make list of all possible 2D states using discretized prices and discretized energy values
        # (eta and battery_capacity are not precisely state variables, they are just stored in the state)
        self.possible_states = []
        for price in self.discrete_prices:
            for energy in self.discrete_energy:
                state = self.model.build_state({'energy_amount': energy,
                                                'price': price,
                                                'eta': self.model.initial_state['eta'],
                                                'battery_capacity': self.model.initial_state['battery_capacity']})
                self.possible_states.append(state)
        # print("possible_states={}".format(self.possible_states))

        time = self.time
        values = []

        while time != -1:
            print("time={}".format(time))
            max_list = {}
            for state in self.possible_states:
                price = state.price
                energy = state.energy_amount
                # print("price={}, energy={}".format(price, energy))
                v_list = []
                for d in self.model.possible_decisions:
                    x = self.model.build_decision(d, energy)
                    contribution = price * (x.sell - x.buy)
                    # print("decision={}, x={}, contribution={}".format(d, x, contribution))
                    sum_w = 0
                    w_index = 0
                    for w in self.discrete_price_changes:
                        f = self.f_p[w_index] if w_index == 0 else self.f_p[w_index] - self.f_p[w_index - 1]
                        next_state = self.state_transition(state, x, w)
                        next_v = values[self.time - time - 1][next_state] if time < self.time \
                            else self.terminal_contribution
                        sum_w += f * next_v
                        # print("w={}, f={}, next_v={}, sum_w={}".format(w, f, next_v, sum_w))
                        w_index += 1
                    # print("w_index={}".format(w_index))
                    v = contribution + sum_w
                    # print("v={}".format(v))
                    v_list.append(v)
                max_value = max(v_list)
                max_list.update({state: max_value})
                # print("max={}, v_list={}, size_max={}".format(max_value, v_list, len(max_list)))
            values.append(max_list)
            time -= 1
        pass
        # print("size_values={}".format(len(values)))
        # print("values={}".format(values))
        return values

    def bellman_3D(self):
        """
        # this function computes the value function using Bellman's equation when the state has 3 dimensions

        :return: list - list of contribution values
        """

        # make list of all possible 3D states using discretized prices and discretized energy values
        self.possible_3D_states = []
        for p in self.discrete_prices:
            for prev_p in self.discrete_prices:
                for energy in self.discrete_energy:
                    state = self.model.build_state({'energy_amount': energy,
                    'price': p,
                    'prev_price': prev_p,
                    'eta': self.model.initial_state['eta'],
                    'battery_capacity': self.model.initial_state['battery_capacity']})
                    self.possible_3D_states.append(state)
        # print("possible_3D_states={}, size={}".format(self.possible_3D_states, len(self.possible_3D_states)))

        time = self.time
        values = []

        while time != -1:
            print("time={}".format(time))
            max_list = {}
            for state in self.possible_3D_states:
                price = state.price
                energy = state.energy_amount
                # print("price={}, energy={}".format(price, energy))
                v_list = []
                for d in self.model.possible_decisions:
                    x = self.model.build_decision(d, energy)
                    contribution = price * (x.sell - x.buy)
                    # print("decision={}, x={}, contribution={}".format(d, x, contribution))
                    sum_w = 0
                    w_index = 0
                    for w in self.discrete_price_changes:
                        f = self.f_p[w_index] if w_index == 0 else self.f_p[w_index] - self.f_p[w_index - 1]
                        next_state = self.state_transition(state, x, w)
                        next_v = values[self.time - time - 1][next_state] if time < self.time \
                            else self.terminal_contribution
                        sum_w += f * next_v
                        # print("w={}, f={}, next_v={}, sum_w={}".format(w, f, next_v, sum_w))
                        w_index += 1
                    # print("w_index={}".format(w_index))
                    v = contribution + sum_w
                    # print("v={}".format(v))
                    v_list.append(v)
                max_value = max(v_list)
                max_list.update({state: max_value})
                # print("max={}, v_list={}, size_max={}".format(max_value, v_list, len(max_list)))
            values.append(max_list)
            time -= 1
        pass
        return values

# unit testing
if __name__ == "__main__":
    # load data from the Excel spreadsheet
    raw_data = pd.read_excel("PJM_Historical_DA_RT_LMPs_05_to_11.xlsx", sheet_name="Raw Data")

    # look at data spanning a week
    t = 0
    stop_time = 50
    data_selection = raw_data.iloc[t:stop_time, 0:5]

    # rename columns to remove spaces (so they can be accessed by name)
    cols = data_selection.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    data_selection.columns = cols

    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')

    # calculate change in price and sort values of change in price in ascending order
    data_selection['Price_Shift'] = data_selection.PJM_RT_LMP.shift(1)
    data_selection['Price_Change'] = data_selection['PJM_RT_LMP'] - data_selection['Price_Shift']
    sort_price_change = data_selection.sort_values('Price_Change')

    # discretize prices by interpolating from cumulative distribution
    xp = sort_by_price['PJM_RT_LMP'].tolist()
    fp = np.arange(sort_price_change['PJM_RT_LMP'].size - 1) / (sort_price_change['PJM_RT_LMP'].size - 1)
    cum_price_fn = np.append(fp, 1)

    # obtain 30 discrete prices
    discrete_increments = np.linspace(0, 1, 30)
    discrete_prices = []
    for i in discrete_increments:
        interpolated_point = np.interp(i, cum_price_fn, xp)
        discrete_prices.append(interpolated_point)

    # discretize change in price and obtain f(p) for each price change
    max_price_change = pd.DataFrame.max(sort_price_change['Price_Change'])
    min_price_change = pd.DataFrame.min(sort_price_change['Price_Change'])
    price_change_range = max_price_change - min_price_change
    price_change_increment = price_change_range / 49
    discrete_price_change = np.arange(min_price_change, max_price_change, price_change_increment)
    discrete_price_change = np.append(discrete_price_change, max_price_change)

    # there are 191 values for price change
    price_changes_sorted = sort_price_change['Price_Change'].tolist()
    # remove the last NaN value
    price_changes_sorted.pop()

    f_p = np.arange(len(price_changes_sorted) - 1) / (len(price_changes_sorted) - 1)
    f_p = np.append(f_p, 1)
    discrete_price_change_pdf = []
    for c in discrete_price_change:
        interpolated_point = np.interp(c, price_changes_sorted, f_p)
        discrete_price_change_pdf.append(interpolated_point)

    # set initial parameter values and create a model
    # 2D states - create a state variable with two dimensions (eta and batt capacity are in the state as requested, but they are not state variables)
    state_variable_2D = ['price', 'energy_amount', 'eta', 'battery_capacity']
    initial_state_2D = {'price': raw_data.iat[t, 4],
                     'energy_amount': 1,
                     'eta': 0.9,
                     'battery_capacity': 20.0}

    # 3D states - create a state variable with three dimensions
    state_variable_3D = ['energy_amount', 'price', 'prev_price', 'eta', 'battery_capacity']
    initial_state_3D = {'energy_amount': 1,
                     'price': raw_data.iat[t + 1, 4],
                     'prev_price': raw_data.iat[t, 4],
                     'eta': 0.9,
                     'battery_capacity': 20.0}

    decision_variable = ['buy', 'hold', 'sell']
    possible_decisions = [{'buy': 1, 'hold': 0, 'sell': 0}, {'buy': 0, 'hold': 0, 'sell': 1},
                          {'buy': 0, 'hold': 1, 'sell': 0}]

    # make list of possible energy amount stored at a time
    min_energy = 0
    max_energy = initial_state_2D['battery_capacity']
    energy_increment = 1
    discrete_energy = np.arange(min_energy, max_energy + 1, energy_increment)

    model_2D = ESM(state_variable_2D, decision_variable, initial_state_2D, raw_data, possible_decisions)
    model_3D = ESM(state_variable_3D, decision_variable, initial_state_3D, raw_data, possible_decisions)

    # create two backward dynamic programming objects - 2D and 3D
    test_2D = BDP(discrete_prices, discrete_energy, sort_price_change['Price_Change'].tolist(), discrete_price_change,
               discrete_price_change_pdf, stop_time, model_2D)
    test_3D = BDP(discrete_prices, discrete_energy, sort_price_change['Price_Change'].tolist(), discrete_price_change,
                  discrete_price_change_pdf, stop_time, model_3D)

    # 2D states - time the process with a 2D state variable
    t0 = time.time()
    value_list = test_2D.bellman_2D()
    t1 = time.time()
    time_elapsed = t1-t0
    print("time_elapsed_2D_model={}s".format(time_elapsed))

    # 3D states - time the process with a 3D state variable (uncomment to run BDP with a 3D state variable)
    t0 = time.time()
    value_list = test_3D.bellman_3D()
    t1 = time.time()
    time_elapsed = t1-t0
    print("time_elapsed_3D_model={}s".format(time_elapsed))