"""
Expected revenue bidding policy

Connor Bridges and Nnaedozie Agbim 2018

Bids sample average of observed revenue

"""

import numpy as np
import math

from .policy import Policy  # this line is needed
from scipy.stats import poisson
from scipy.stats import binom


# noinspection SpellCheckingInspection
class Policy_ConnorDozie_LAPS(Policy):

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
        self.tau = 3  # tau is the number of lookahead steps taken, tuned
        self.t = 0  # current iteration
        self.profit = {}
        for attr in all_attrs:
            # from analysis of average values obtained
            self.gamma[attr] = 0.25
            self.lamda[attr] = 100
            self.eta[attr] = 0.15
            self.returns[attr] = 50
            # prior probabilities
            self.prob[attr] = [1 / self.K] * self.K
            # precompute the profit for each bid i under each curve j
            self.profit[attr] = [[0]*self.K for _ in range(len(self.bid_space))]
            for i in range(len(self.bid_space)):
                for j in range(self.K):
                    b = self.bid_space[i]
                    x = (b,)
                    theta = self.theta[j]
                    alpha = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x) - theta[2])))
                    self.profit[attr][i][j] = alpha*self.lamda[attr]*self.gamma[attr]*((self.eta[attr]*self.returns[attr])-b)


    def bid(self, attr):
        """
        finds a bid that is closest to the revenue sample mean of auctions with the given attr

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: the decision of the policy
        """

        # mu vector at time t
        mu_t = [0] * len(self.bid_space)
        for i in range(len(self.bid_space)):
            # computing mu for bid b at time t
            b = self.bid_space[i]
            x = (b,)
            for j in range(self.K):
                theta = self.theta[j]
                alpha_j = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x) - theta[2])))
                f = alpha_j * self.lamda[attr] * self.gamma[attr] * ((self.eta[attr] * self.returns[attr]) - b)
                mu_t[i] += self.prob[attr][j] * f

        # compute best b
        values = [0] * len(self.bid_space)
        for i in range(len(self.bid_space)):
            values[i] = mu_t[i] + ((self.max_t - self.t) / self.tau) * self.kg(attr, self.bid_space[i], mu_t)
        best_bid = np.argmax(values)
        best_bid = self.bid_space[best_bid]
        return best_bid

    def kg(self, attr, bid, mu_t):
        # x vector to compute KG for
        x_kg = (bid,)
        alpha_kg = [0] * self.K
        sample_size = 30
        for j in range(self.K):
            theta = self.theta[j]
            alpha_kg[j] = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x_kg) - theta[2])))
        # mu vector at time t + tau
        mu_t_tau = [0] * len(self.bid_space)
        # max_arrivals = 5 * self.tau  # cut off for number of possible poisson arrivals in the time period
        # max_mu_t_tau = 0  # this is the Emax term in KG (note that it is a scalar)
        kg_sample = [0]*sample_size
        alpha = [0] * self.K
        p_win = 0
        for j in range(self.K):
            theta = self.theta[j]
            alpha[j] = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x_kg) - theta[2])))
            p_win += self.prob[attr][j]*alpha[j]
        for l in range(sample_size):
            a = np.random.poisson(self.lamda[attr]*self.tau)
            imp = np.random.binomial(a, p_win)
            # p_j^(t+tau)
            p_belief = [0] * self.K
            for j in range(self.K):
                # probability of a arrivals and imp impressions
                # compute p_j^(t+tau)
                p_belief[j] = np.power(alpha[j], imp) * np.power(1 - alpha[j], a - imp) * self.prob[attr][j]
            p_belief = np.divide(p_belief, np.sum(p_belief))
            for j in range(len(self.bid_space)):
                mu_t_tau[j] = np.sum(np.multiply(p_belief, self.profit[attr][j]))  # recall profit[attr][j] is a vector of profits under different curves
            kg_sample[l] = np.max(mu_t_tau) - np.max(mu_t)
        kg_val = np.sum(kg_sample)/sample_size # approximate kg by a sampled average


        #for a in range(max_arrivals + 1):
        #    for imp in range(a + 1):
        #        # p_j^(t+tau)
        #        p_belief = [0] * self.K
        #
        #        # probability of a arrivals and imp impressions
        #        p = 0
        #        for j in range(self.K):
        #            # compute p_j^(t+tau)
        #            p_belief[j] = np.power(alpha[j], imp) * np.power(1 - alpha[j], a - imp)*self.prob[j]
        #            p += self.prob[j] * poisson.pmf(a, self.tau * self.lamda[attr]) * binom.pmf(imp, a, alpha_kg[j])
        #        p_belief = np.divide(p_belief, np.sum(p_belief))
        #        for j in range(len(self.bid_space)):
        #            mu_t_tau[j] = np.dot(p_belief, self.profit[attr][j]) # recall profit[attr][j] is a vector of profits under different curves
        #        max_mu_t_tau += np.max(mu_t_tau)*p
        #kg_val = max_mu_t_tau - np.max(mu_t)
        return kg_val

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
        for result in info:
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
            if result['revenue_per_conversion'] != "":
                returns_old = self.returns[attr]
                self.returns[attr] = (weight_old * returns_old) + \
                           (weight_new * (result['revenue_per_conversion'] * result['num_conversion']))

            # update probabilities
            x = (result['your_bid'],)
            for i in range(self.K):
                theta = self.theta[i]
                alpha_i = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x) - theta[2])))
                self.prob[attr][i] = np.power(alpha_i, result['num_impression']) * \
                               np.power(1 - alpha_i, result['num_auct'] - result['num_impression'])
                # this is the binomial formula - the nCk cancels out in the numerator and denominator
            self.prob[attr] = np.divide(self.prob[attr], np.sum(self.prob[attr]))  # normalize this probability vector

            # updating the profit matrix
            for i in range(len(self.bid_space)):
                for j in range(self.K):
                    b = self.bid_space[i]
                    x = (b,)
                    theta = self.theta[j]
                    alpha = 1 / (1 + np.exp(-theta[0] * (np.dot(theta[1], x) - theta[2])))
                    self.profit[attr][i][j] = alpha * self.lamda[attr] * self.gamma[attr] * ((self.eta[attr] * self.returns[attr]) - b)

        self.t += 1
        return True
