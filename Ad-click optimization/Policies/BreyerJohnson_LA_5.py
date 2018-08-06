"""
Knowledge Gradient Beta Bernoulli



"""


import numpy as np
from numpy import matrix


from .policy import Policy    # this line is needed


class Policy_BreyerJohnson_LA_5(Policy):

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

        self.initialize_belief()

    def initialize_belief(self):
        """
        initial average values are set up.


        :return:
        """

        self.alpha = np.array([[0, 1, 0, 3, 4, 4, 0, 4, 1, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1, 3, 0],
                      [0, 1, 3, 4, 7, 2, 3, 2, 6, 3],
                      [0, 0, 0, 0, 1, 3, 2, 0, 1, 1],
                      [0, 0, 0, 1, 0, 1, 1, 3, 0, 2],
                      [0, 0, 1, 5, 6, 2, 3, 3, 4, 4],
                      [0, 4, 4, 6, 5, 1, 5, 4, 5, 5],
                      [0, 1, 1, 3, 0, 2, 0, 0, 6, 5],
                      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 1, 3, 1, 1, 2, 4, 3],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 3, 9, 3, 6, 3, 7, 1],
                      [0, 0, 0, 0, 0, 0, 1, 2, 0, 1],
                      [0, 1, 3, 0, 3, 5, 2, 4, 7, 5]])

        self.beta = np.array([[123, 113, 116,  92,  86, 118,  94, 118, 114, 125],
                     [100, 95, 100, 92, 100, 102,  99, 104,  94, 109],
                     [106, 91, 100, 93, 89, 103,  92, 101, 93, 101],
                     [104, 108, 104, 98, 90, 119, 94,  93, 94, 88],
                     [94, 101, 82, 94, 103, 103, 110, 100, 100, 105],
                     [93, 109, 105,  88, 98,  97,  95,  95, 104,  88],
                     [117,  95,  98,  94, 101, 84, 95,  92,  88, 97],
                     [81, 101,  94, 97, 109,  96,  84, 105,  97, 118],
                     [98, 85, 99, 101, 108,  98, 102,  79, 120, 107],
                     [110,  94,  93, 96, 96,  95,  92,  95, 114, 100],
                     [0, 0,  0,  0,  0,  0,  0,  0,  0,  0],
                     [101,  96, 114,  80,  81,  98, 96, 102, 96, 103],
                     [108, 111,  90, 97,  99, 102,  98,  93,  94, 104],
                     [123, 113, 116, 92, 86, 118, 94, 118, 114, 125]])

        self.var = np.zeros((14, 10))

        for i in range(14):
            for j in range(10):
                prob = self.alpha[i, j] / (self.alpha[i, j] + self.beta[i, j])
                self.var[i, j] = prob * (1-prob)

    def bid(self, attr):
        """
        finds a bid that is closest to the revenue sample mean of auctions with the given attr

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """

        k = 10
        C_max = np.zeros(10)
        a_ix = self.attrs.index(attr)
        prob_array = self.alpha[a_ix, :] / (self.alpha[a_ix, :] + self.beta[a_ix, :])

        for i in range(len(C_max)):
            temp = prob_array
            temp[i] = 0
            C_max[i] = np.amax(temp)

        curr = self.alpha[a_ix, :] / (self.alpha[a_ix, :] + self.beta[a_ix, :])
        succ = (self.alpha[a_ix, :] + 1) / (self.alpha[a_ix, :] + self.beta[a_ix, :] + 1)
        fail = (self.alpha[a_ix, :]) / (self.alpha[a_ix, :] + self.beta[a_ix, :] + 1)

        vkg = np.zeros(10)
        for i in range(10):
            if curr[i] < C_max[i]:
                if C_max[i] < succ[i]:
                    vkg[i] = curr[i] * (succ[i] - C_max[i])
            if fail[i] < C_max[i]:
                if C_max[i] < curr[i]:
                    vkg[i] = (1 / curr[i]) * (C_max[i] - fail[i])
            else:
                vkg[i] = 0

        onlineKG = np.zeros(10)


        theta1 = 5
        for j in range(10):
            onlineKG[j] = curr[j] + (theta1 * vkg[i])

        decision = np.argmax(onlineKG)

        overbid = np.random.rand()
        decision = decision - round(overbid, 1)
        if (decision > 9.9):
            decision = 9.9
        if (decision < 0):
            decision = 0

        return decision

    def learn(self, info):
        """
        learns from auctions results

        In this sample policy, it learns by keeping track of sample averages of revenue from auctions of each attribute.
        If 'revenue_per_conversion' is empty, that means I did not get any conversions in those auctions. I ignore them.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """

        for result in info:

            x = int(result['your_bid'])
            attr = result['attr']
            a_ix = self.attrs.index(attr)
            if result['num_click'] == '':
                clicks = 0
            else:
                clicks = int(result['num_click'])
            if result['cost_per_click'] == '':
                cost = 0
            else:
                cost = int(result['cost_per_click'])
            cost = cost * clicks
            if result['revenue_per_conversion'] == '':
                revenue = 0
            else:
                revenue = int(result['revenue_per_conversion'])
            profit = revenue - cost

            if profit <= 0:
                self.beta[a_ix, x] = self.beta[a_ix, x] + 1
            else:
                self.alpha[a_ix, x] = self.alpha[a_ix, x] + 1

        return True

