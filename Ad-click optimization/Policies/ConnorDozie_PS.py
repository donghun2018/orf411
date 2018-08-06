"""
Expected revenue bidding policy

Connor Bridges and Nnaedozie Agbim 2018

Bids sample average of observed revenue

"""

import numpy as np

from .policy import Policy  # this line is needed
from scipy.stats import poisson
from scipy.stats import binom


# noinspection SpellCheckingInspection
class Policy_ConnorDozie_PS(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12345):
        """
        initializes policy class.

        Note that the first line must be that super().__init__ ...

        Please use self.prng. instead of np.random. if you want to give randomness to your policy

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)

        # initialize parameters
        self.lamda = {}    # average number of auctions per period
        self.gamma = {}    # average click through rate
        self.eta = {}      # average sale conversion probability following click through
        self.returns = {}  # average revenue per sale
        self.prob = {}     # probability of each belief
        self.theta = [[0.5, 0.925, 7], [0.7, 0.95, 6], [0.8, 0.9, 7], [1.1, 1, 5]]  # prior beliefs, for k = 4 (4 curves)
        self.K = len(self.theta)  # K is the number of vectors above
        self.boltzmann = 0.5  # Boltzmann parameter - low to start
        for attr in all_attrs:
            self.gamma[attr] = 0.25
            self.lamda[attr] = 100
            self.eta[attr] = 0.15
            self.returns[attr] = 50
            self.prob[attr] = [1 / self.K] * self.K


    def bid(self, attr):
        """
        finds a bid that is closest to the revenue sample mean of auctions with the given attr

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: the decision of the policy
        """

        # Boltzmann probability vector
        b_prob = [0] * len(self.bid_space)
        mu_vector = [0] * len(self.bid_space)
        for i in range(len(self.bid_space)):
            # computing mu for bid b at time t
            b = self.bid_space[i]
            x = (b,)
            for j in range(self.K):
                theta = self.theta[j]
                alpha_j = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x) - theta[2])))
                f = alpha_j * self.lamda[attr] * self.gamma[attr] * ((self.eta[attr] * self.returns[attr]) - b)
                mu_vector[i] += self.prob[attr][j] * f
        # prevent overflow
        max_mu = np.max(mu_vector)
        for i in range(len(self.bid_space)):
            b_prob[i] = np.exp((mu_vector[i] - max_mu) * self.boltzmann)
        # normalize vector
        b_prob = np.divide(b_prob, np.sum(b_prob))
        u = np.random.uniform()
        cum_prob = b_prob[0]
        index = 0
        while cum_prob <= u:
            cum_prob += b_prob[index + 1]
            index += 1
        return self.bid_space[index]

    def learn(self, info):
        """
        learns from auctions results

        In this sample policy, it learns by keeping track of sample averages of revenue from auctions of each attribute.
        If 'revenue_per_conversion' is empty, that means I did not get any conversions in those auctions. I ignore them.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        bid_win_p = [0] * self.K  # initialize average probability of winning an auction under belief j to 0

        # updating our point estimates
        profitable = 0
        total = 0
        t = 0
        for result in info:
            if t == 0:
                t = result['iter']
            # get result attributes
            attr = result['attr']
            # compute weights
            weight_old = (1 - (1 / (result['iter'] + 1)))
            weight_new = (1 / (result['iter'] + 1))
            # update lamda
            lamda_old = self.lamda[attr]
            self.lamda[attr] = (weight_old * lamda_old) + (weight_new * result['num_auct'])
            # update gamma
            if result['num_impression'] != 0:
                gamma_old = self.gamma[attr]
                self.gamma[attr] = (weight_old * gamma_old) + \
                               (weight_new * (result['num_click'] / result['num_impression']))
            # update eta
            if result['num_click'] != 0:
                eta_old = self.eta[attr]
                self.eta[attr] = (weight_old * eta_old) + (weight_new * (result['num_conversion'] / result['num_click']))
            # update returns
            if result['revenue_per_conversion']!="":
                total = total + 1
                returns_old = self.returns[attr]
                self.returns[attr] = (weight_old * returns_old) + \
                           (weight_new * (result['revenue_per_conversion'] * result['num_conversion']))
                if result['revenue_per_conversion']>0:
                    profitable = profitable + 1

            # update probabilities
            x = (result['your_bid'],)
            for i in range(self.K):
                theta = self.theta[i]
                alpha_i = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x) - theta[2])))
                self.prob[attr][i] = np.power(alpha_i, result['num_impression']) * \
                               np.power(1 - alpha_i, result['num_auct'] - result['num_impression'])
                # this is the binomial formula - the nCk cancels out in the numerator and denominator
            self.prob[attr] = np.divide(self.prob[attr], np.sum(self.prob[attr]))  # normalize this probability vector

        # allow for near-pure exploration initially, then revert to 'normal' Boltzmann
        if t == 10: self.boltzmann = 2
        # dynamically update Boltzmann parameter
        if t/self.max_t > 0.60:
            if total > 0:
                if profitable/total < t/self.max_t:
                    self.boltzmann = self.boltzmann + 0.1
        return True
