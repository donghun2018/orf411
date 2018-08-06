"""
Energy storage policy class

"""
from collections import namedtuple
import pandas as pd
import numpy as np
from EnergyStorageModel import EnergyStorageModel as ESM
import matplotlib.pyplot as plt
from copy import copy

class EnergyStoragePolicy():
    """
    Base class for decision policy
    """

    def __init__(self, model, policy_names):
        """
        Initializes the policy

        :param model: EnergyStorageModel - the model that the policy is being implemented on
        :param policy_names: list(str) - list of policies
        """

        self.model = model
        self.policy_names = policy_names
        self.Policy = namedtuple('Policy', policy_names)

    def build_policy(self, info):
        """
        this function builds the policies depending on the parameters provided

        :param info: dict - contains all policy information
        :return: namedtuple - a policy object
        """
        return self.Policy(*[info[k] for k in self.policy_names])

    def buy_low_sell_high_policy(self, state, info_tuple):
        """
        this function implements the buy low, sell high policy for the ESM

        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """
        lower_limit = info_tuple[0]
        upper_limit = info_tuple[1]
        if state.price <= lower_limit:
            new_decision = self.model.possible_decisions[0]
        elif state.price >= upper_limit:
            new_decision = self.model.possible_decisions[1]
        else:
            new_decision = self.model.possible_decisions[2]
        return new_decision

    def run_policy(self, policy_info, policy, stop_time):
        """
        this function runs the model with a selected policy

        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param stop_time: float - stop time
        :return: float - calculated contribution
        """
        time = 0
        model_copy = copy(self.model)

        while time != stop_time + 1:
            # build decision policy
            p = self.build_policy(policy_info)

            # make decision based on chosen policy
            if policy == "buy_low_sell_high":
                decision = self.buy_low_sell_high_policy(model_copy.state, p.buy_low_sell_high)

            x = model_copy.build_decision(decision, model_copy.state.energy_amount)
            print("time={}, obj={}, state.energy_amount={}, state.price={}, x={}".format(time, model_copy.objective,
                                                                                      model_copy.state.energy_amount,
                                                                                      model_copy.state.price, x))
            # step the model forward one iteration
            model_copy.step(time, x)
            # increment time
            time += 1
        contribution = model_copy.objective
        return contribution

    def vary_theta(self, policy_info, policy, stop_time, theta_values):
        """
        this function calculates the contribution for each theta value in a list

        :param policy_info: dict - dictionary of policies and their associated parameters
        :param policy: str - the name of the chosen policy
        :param stop_time: float - stop time
        :param theta_values: list - list of all possible thetas to be tested
        :return: list - list of contribution values corresponding to each theta
        """

        contribution_values = []
        for theta in theta_values:
            t = stop_time
            policy_dict = policy_info.copy()
            policy_dict.update({'buy_low_sell_high': theta})
            print("policy_dict={}".format(policy_dict))
            contribution = self.run_policy(policy_dict, policy, t)
            contribution_values.append(contribution)
        return contribution_values

    def grid_search_theta_values(self, buy_min, buy_max, sell_min, sell_max, increment_size):
        """
        this function gives a list of theta values needed to run a full grid search

        :param buy_min: the minimum value/lower bound of theta_buy
        :param buy_max: the maximum value/upper bound of theta_buy
        :param sell_min: the minimum value/lower bound of theta_sell
        :param sell_max: the maximum value/upper bound of theta_sell
        :param increment_size: the increment size over the range of theta values
        :return: list - list of theta values
        """
        theta_buy_values = np.linspace(buy_min, buy_max, (buy_max - buy_min)/increment_size + 1)
        theta_sell_values = np.linspace(sell_min, sell_max, (sell_max - sell_min)/increment_size + 1)

        theta_values = []
        for x in theta_buy_values:
            for y in theta_sell_values:
                theta = (x, y)
                theta_values.append(theta)
        return theta_values, theta_buy_values, theta_sell_values

    def theta_buy_plot_values(self, buy_min, buy_max, increment_size, theta_sell_values):
        """
        this function gives a list of theta values needed to plot performance as a function of the buy value
        for selected theta_sell values

        :param buy_min: the minimum value/lower bound of theta_buy
        :param buy_max: the maximum value/upper bound of theta_buy
        :param increment_size: the increment size over the range of theta values
        :param theta_sell_values: list of theta_sell values (from an Excel spreadsheet)
        :return: list - list of theta values
        """
        theta_buy_values = np.linspace(buy_min, buy_max, (buy_max - buy_min)/increment_size + 1)
        theta_values = []
        for y in theta_sell_values:
            for x in theta_buy_values:
                theta = (x, y)
                theta_values.append(theta)
        return theta_values, theta_buy_values, theta_sell_values

    def plot_heat_map(self, contribution_values, theta_buy_values, theta_sell_values):
        """
        this function plots a heat map

        :param contribution_values: list - list of contribution values
        :param theta_buy_values: list - list of theta_buy_values
        :param theta_sell_values: list - list of theta_sell_values
        :return: none (plots a heat map)
        """
        contributions = np.array(contribution_values)
        increment_count = len(theta_buy_values)
        contributions = np.reshape(contributions, (-1, increment_count))

        fig, ax = plt.subplots()
        im = ax.imshow(contributions, cmap='hot')
        # create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        # we want to show all ticks...
        ax.set_xticks(np.arange(len(theta_buy_values)))
        ax.set_yticks(np.arange(len(theta_sell_values)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(theta_buy_values)
        ax.set_yticklabels(theta_sell_values)
        # rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_title("Heatmap of contribution values across different values of theta")
        fig.tight_layout()
        plt.show()
        return True

    def plot_theta_buy(self, contribution_values, theta_buy_values, theta_sell_values):
        """
        this function plots performance as a function of the theta_buy value for selected theta_sell values

        :param contribution_values: list - list of contribution values
        :param theta_buy_values: list - list of theta_buy_values
        :param theta_sell_values: list - list of theta_sell_values
        :return: none (plots line graphs)
        """
        contributions = np.array(contribution_values)
        increment_count = len(theta_buy_values)
        contributions = np.reshape(contributions, (-1, increment_count))

        legend_labels = ['theta_sell = ' + str(theta_sell_values[0]), 'theta_sell = ' + str(theta_sell_values[1]),
        'theta_sell = ' + str(theta_sell_values[2]), 'theta_sell = ' + str(theta_sell_values[3]),
        'theta_sell = ' + str(theta_sell_values[4])]

        # plot contribution values for 5 different values of theta_sell over all values of theta_buy
        graph_1 = plt.plot(theta_buy_values, contributions[0], 'r', label = legend_labels[0])
        graph_2 = plt.plot(theta_buy_values, contributions[1], 'g', label = legend_labels[1])
        graph_3 = plt.plot(theta_buy_values, contributions[2], 'b', label = legend_labels[2])
        graph_4 = plt.plot(theta_buy_values, contributions[3], 'c', label = legend_labels[3])
        graph_5 = plt.plot(theta_buy_values, contributions[4], 'm', label = legend_labels[4])

        plt.legend()
        plt.title("Contribution values for 5 different values of theta_sell")
        plt.show()
        return True

# unit testing
if __name__ == "__main__":
    # load energy price data from the Excel spreadsheet
    raw_data = pd.read_excel("PJM_Historical_DA_RT_LMPs_05_to_11.xlsx", sheet_name="Raw Data")

    # look at data spanning a week
    t = 0
    stop_time = 191
    data_selection = raw_data.iloc[t:stop_time, 0:5]

    # rename columns to remove spaces (otherwise we can't access them)
    cols = data_selection.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, str) else x)
    data_selection.columns = cols

    # sort prices in ascending order
    sort_by_price = data_selection.sort_values('PJM_RT_LMP')
    print(sort_by_price)

    # create a model and a policy
    policy_names = ['buy_low_sell_high']
    state_variable = ['price', 'energy_amount', 'eta', 'battery_capacity']
    initial_state = {'price': raw_data.iat[t, 4],
                     'energy_amount': 1,
                     'eta': 0.9,
                     'battery_capacity': 20.0}
    decision_variable = ['buy', 'hold', 'sell']
    possible_decisions = [{'buy': 1, 'hold': 0, 'sell': 0}, {'buy': 0, 'hold': 0, 'sell': 1},
                          {'buy': 0, 'hold': 1, 'sell': 0}]
    M = ESM(state_variable, decision_variable, initial_state, raw_data, possible_decisions)
    P = EnergyStoragePolicy(M, policy_names)

    # read in policy parameters from an Excel spreadsheet, "energy_storage_policy_parameters.xlsx"
    sheet1 = pd.read_excel("energy_storage_policy_parameters.xlsx", sheet_name="Sheet1")
    sheet2 = pd.read_excel("energy_storage_policy_parameters.xlsx", sheet_name="Sheet2")
    sheet3 = pd.read_excel("energy_storage_policy_parameters.xlsx", sheet_name="Sheet3")
    sheet4 = pd.read_excel("energy_storage_policy_parameters.xlsx", sheet_name="Sheet4")
    params = zip(sheet1['param1'], sheet1['param2'])
    param_list = list(params)
    theta_sell_values = list(sheet2['theta_sell_values'])
    policy_info = {'buy_low_sell_high': param_list[0]}

    # an example of running the policy for the chosen time period (one week, in this case)
    P.run_policy(policy_info, "buy_low_sell_high", stop_time)

    # below are two possible ways to visualize the results: i) a heat map showing the results of a full grid search and
    # ii) a plot of performance as a function of the buy value for 5 selected theta_sell values

    # i) do full grid search across values of theta_buy and theta_sell
    # discretize theta_buy and theta_sell into 5% increments
    max_price = pd.DataFrame.max(sort_by_price['PJM_RT_LMP'])
    min_price = pd.DataFrame.min(sort_by_price['PJM_RT_LMP'])
    increment_size = (max_price - min_price) / 20

    # obtain the theta values to carry out a full grid search
    grid_search_theta_values = P.grid_search_theta_values(min_price, max_price, min_price, max_price, increment_size)
    # alternatively, change parameter values in Sheet3 of energy_storage_policy_parameters.xlsx
    # grid_search_theta_values = P.grid_search_theta_values(sheet3['low_min'], sheet3['low_max'], sheet3['high_min'],
    #                                                          sheet3['high_max'], sheet3['increment_size'])

    # use those theta values to calculate corresponding contribution values
    contribution_values_1 = P.vary_theta(policy_info, "buy_low_sell_high", stop_time, grid_search_theta_values[0])

    # plot those contribution values on a heat map, with theta_buy on the horizontal axis and theta_sell on the
    # vertical axis
    P.plot_heat_map(contribution_values_1, grid_search_theta_values[1], grid_search_theta_values[2])


    # ii) obtain the theta values to plot performance as a function of the buy value for selected theta_sell values
    theta_buy_plot_values = P.theta_buy_plot_values(min_price, max_price, increment_size, theta_sell_values)
    # alternatively, change parameter values in Sheet4 of energy_storage_policy_parameters.xlsx
    # theta_buy_plot_values = P.theta_buy_plot_values (sheet4['low_min'], sheet4['low_max'], sheet4['increment_size'])

    # use those theta values to calculate corresponding contribution values
    contribution_values_2 = P.vary_theta(policy_info, "buy_low_sell_high", stop_time, theta_buy_plot_values[0])

    # plot performance as a function of the theta_buy value for selected theta_sell values
    P.plot_theta_buy(contribution_values_2, theta_buy_plot_values[1], theta_buy_plot_values[2])