"""
Bidding policy base class

Adapted from code by Larry Bao and Kate Wang (ORF 418 Spring 2018)

A policy that implements Boltzmann for the ad-click problem.
"""
import numpy as np
from .policy import Policy    # this line is needed

class Policy_AdClickPolicy(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=41023):
        """
        initializes policy base class.

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """

        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)
        self.bid_price = {}
        self.alpha = {} # probabilities of clicking
        self.beta = {} # probability of converting
        self.gamma = {} # revenue per conversion
        self.lamb = {} # lambda - number of auctions
        self.cumClicks = {}
        # tunable parameters - will read in values from Excel file "Parameter_Values.xlsx"
        self.theta_b
        self.w

        # exogenous information
        self.A_t = {} # number of auctions
        self.I_t = {} # number of impressions
        self.K_t = {} # number of clicks
        self.C_t = {} # number of conversions
        self.R_t = {} # revenue / conversions

        # sets of thetas
        self.theta0 = [0.4, 0.5, 0.6, 0.7, 0.8, 0.3, 0.35, 0.5]
        self.theta1 = [0.05, 0.07, 0.09, 0.1, 0.11, 0.1, 0.09, 0.09]
        self.theta2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.05, 0.03]
        self.theta3 = [6, 4.5, 5, 5.5, 6.5, 7, 6.5, 5]

        # assign the thetas in the lists above with equal probabilities of 0.125 (since there are 8 theta values)
        self.prob_win = [1/len(self.theta0)] * len(self.theta0)

        # initialize arrays
        for attr in all_attrs:
            self.bid_price[attr] = self.prng.choice(self.bid_space)
            self.A_t[attr] = 0
            self.I_t[attr] = 0
            self.K_t[attr] = 0
            self.C_t[attr] = 0
            self.R_t[attr] = 0
            self.alpha[attr] = 0.3
            self.lamb[attr] = 110
            self.beta[attr] = 0.1
            self.gamma[attr] = 35
            self.cumClicks[attr] = 0

    def bid(self, attr):
        """
        returns a random bid, regardless of attribute

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        # initialize mu, an array of zeros with the length of self.bid_space (100)
        mu = np.zeros(len(self.bid_space))

        # j ranges from 0 to 99
        for j in range(len(mu)):
            # i ranges from 0 to 7
            for i in range(len(self.prob_win)):
                # calculate mu values
                mu[j] = mu[j] + self.prob_win[i] * \
                        (self.gamma[attr] * self.beta[attr] - self.bid_space[j]) * \
                        self.lamb[attr] * self.alpha[attr] * self.p_x(self.bid_space[j], i, attr)

        rand = self.prng.uniform(0, 1)

        x = 0
        px = np.exp(np.multiply(self.theta_b, mu))
        px = np.divide(px, np.sum(px))
        px = np.cumsum(px)

        # randomly choose index of bid_space
        for i in range(len(mu)):
            if px[i] >= rand:
                x = i
                break

        return self.bid_space[x]

    # function to calculate p_x
    def p_x(self, x, j, attr):
        # attr needs to be replaced with the features in attr
        y = 1 / (1 + np.exp(-(self.theta0[j] * x + self.theta1[j] * attr[0] + self.theta2[j] * attr[1]) +
                            self.theta3[j]))
        return y

    # function to calculate probability of winning a bid, p_w
    def p_w(self, x, j, attr):
        y = np.power(self.p_x(x, j, attr), self.I_t[attr]) * np.power(1 - self.p_x(x, j, attr), self.A_t[attr] -
                                                                      self.I_t[attr])
        return y

    def learn(self, info):
        """
        learns from auctions results

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        # update number of auctions
        for result in info:
            attr = result['attr']
            self.A_t[attr] = result['num_auct']
            self.lamb[attr] = self.lamb[attr] * self.w + self.A_t[attr] * (1 - self.w)

            # update probability of winning a bid
            for j in range(len(self.theta0)):
                Pw = self.p_w(result['your_bid'], j, attr)
                self.prob_win[j] = self.prob_win[j] * Pw
            self.prob_win = np.divide(self.prob_win, np.sum(self.prob_win))

            # case where we win the bid and get clicks - update number of impressions, number of clicks,
            # number of conversions, revenue per conversion, probability of conversion, cumulative clicks and
            # probability of clicking
            if result['num_click'] > 0:
                self.I_t[attr] = result['num_impression']
                self.K_t[attr] = result['num_click']
                self.C_t[attr] = result['num_conversion']
                self.R_t[attr] = result['revenue_per_conversion']
                self.alpha[attr] = self.alpha[attr] * self.w + self.K_t[attr] / self.I_t[attr] * (1 - self.w)
                self.beta[attr] = (self.beta[attr] * self.cumClicks[attr] + self.C_t[attr]) / \
                                  (self.cumClicks[attr] + self.K_t[attr])
                self.cumClicks[attr] = self.cumClicks[attr] + result['num_click']

                # case where clicks converted into purchases
                if result['num_conversion'] > 0:
                    self.gamma[attr] = self.gamma[attr] * self.w + self.R_t[attr] * (1 - self.w)

            # case where we lose the bid - zero the number of impressions
            else:
                self.I_t[attr] = 0

        #print('lambda {} beta {} gamma {} prob_win {}'.format(self.lamb[attr], self.beta[attr], self.gamma[attr], self.prob_win))
        return True



