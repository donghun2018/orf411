import numpy as np
import pandas as pd

class TimeSeries():
    """
    Base class for time series model
    """

    def __init__(self, initial_theta, initial_b, prices):

        """
        Initializes the policy

        :param initial_theta: initial vector of coefficients
        :param initial_b: initial matrix B_0
        :param prices: first three prices observed
        """

        self.initial_theta = initial_theta
        self.initial_b = initial_b
        self.prices = prices
        self.initial_time = 0

    def phi(self, time):
        """
        function that gives phi_time - a 3-by-1 matrix

        :param time: int
        :return: phi_time - a 3-by-1 matrix
        """
        if time < self.initial_time:
            raise ValueError('Time cannot take a negative value')
        elif time == self.initial_time:
            phi = np.matrix([self.prices[time], self.prices[time], self.prices[time]])
        elif time == self.initial_time + 1:
            phi = np.matrix([self.prices[time], self.prices[time - 1], self.prices[time - 1]])
        else:
            phi = np.matrix([self.prices[time], self.prices[time - 1], self.prices[time - 2]])
        return np.transpose(phi)

    def gamma(self, time, b):
        """
        function that gives gamma_time - a scalar

        :param time: int
        :param b: a matrix
        :return: gamma_time - a scalar
        """
        if time < self.initial_time:
            raise ValueError('Time cannot take a negative value')
        else:
            phi = self.phi(time)
            gamma = 1 + np.matmul(np.matmul(np.transpose(phi), b), phi)
        return gamma

    def recursive_matrix(self, time):
        """
        function that gives the matrix B_time - a 3-by-3 matrix

        :param time: int
        :return: matrix B_time - a 3-by-3 matrix
        """

        phi = self.phi(time)

        if time < self.initial_time:
            raise ValueError('Time cannot take a negative value')

        elif time == self.initial_time:
            return self.initial_b

        else:
            previous_b = self.recursive_matrix(time - 1)
            gamma = self.gamma(time, previous_b)
            b_n = previous_b - \
                  (np.matmul(np.matmul(np.matmul(previous_b, phi), np.transpose(phi)),
                             previous_b)) / gamma
            return b_n
        pass

    def h_matrix(self, time):
        """
        function that gives the matrix H_time - a 3-by-3 matrix

        :param time: int
        :return: matrix H_time - a 3-by-3 matrix
        """
        if time == self.initial_time:
            b = self.initial_b
        else:
            b = self.recursive_matrix(time - 1)
        gamma = self.gamma(time, b)
        h = b / gamma
        return h

    def error(self, time):
        """
        function that gives error_time - a 1-by-1 matrix

        :param time: int
        :return: error_time - a 1-by-1 matrix
        """
        if time < self.initial_time:
            raise ValueError('We need to begin indexing time from time = 3')

        else:
            phi = self.phi(time - 1)
            theta = self.theta(time - 1)
            error = np.matmul(np.transpose(theta), phi) - np.matrix(self.prices[time])

        return error

    def theta(self, time):
        """
        function that gives theta_time - a 3-by-1 matrix

        :param time: int
        :return: theta_time - a 3-by-1 matrix
        """
        if time < self.initial_time:
            raise ValueError('Time cannot take a negative value')

        elif time == self.initial_time:
            return self.initial_theta

        elif time > self.initial_time:
            prev_theta = self.theta(time - 1)
            phi = self.phi(time - 1)
            h = self.h_matrix(time - 1)
            error = self.error(time)
            new_theta = prev_theta - \
                        np.matmul(np.matmul(h, phi), error)

            return new_theta

    def next_price(self, time):
        """
        this function returns the next price p_t+1 as a function of the last three prices

        :param time: int
        :return: float - next price p_t+1
        """
        theta = self.theta(time)
        phi = self.phi(time)
        error = self.error(time).item(0)
        # print("phi={}, theta={}, error={}".format(phi, theta, error))
        next_price = (np.matmul(np.transpose(theta), phi) + error).item(0)
        self.prices[time + 1] = next_price
        return next_price

# unit testing
if __name__ == "__main__":

    # load data from the Excel spreadsheets
    file = 'PJM_Historical_DA_RT_LMPs_05_to_11.xlsx'
    xl = pd.ExcelFile(file)
    raw_data = xl.parse('Raw Data')
    sheet5 = pd.read_excel("energy_storage_policy_parameters.xlsx", sheet_name="Sheet5")
    sheet6 = pd.read_excel("energy_storage_policy_parameters.xlsx", sheet_name="Sheet6")
    init_theta_values = list(sheet5['init_theta'])
    init_b_values = list(sheet6['init_b'])
    init_b_values = np.array(init_b_values)
    init_b_values = np.reshape(init_b_values, (-1, 3))

    # set start and stop times
    t = 0
    stop_time = 192

    # initialize the time series model
    init_theta = np.matrix([init_theta_values])
    initial_theta = np.transpose(init_theta)
    initial_b = np.matrix(init_b_values)
    initial_prices = {0: raw_data.iat[t, 4], 1: raw_data.iat[t + 1, 4], 2: raw_data.iat[t + 2, 4]}

    time_series = TimeSeries(initial_theta, initial_b, initial_prices)

    # run the time series model for some time period
    time = 2
    while time != 15:
        time_series.next_price(time)
        time += 1
    print(time_series.prices)
