""" Bidding Policy base class Donghun Lee 2018 All policy classes inherit this class. """

import numpy as np
from .policy import Policy    # this line is needed


class Policy_WagnerMMegwa_interval_est(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12345):
        """
        initializes policy base class.
        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)
        
        self.initialize_belief_model()


    def initialize_belief_model(self):
        #Mean cost of each attribute class. Var of cost for each attribute class
        
        self.mean = np.array([[1.778], [3.408],[0.842],[3.159],[3.126],[2.097],[1.002],[1.444],[3.492],[2.275],[4.112],[0.978],[3.510],[0.795]])
        self.var = np.array([[0.708],[3.846],[0.172],[3.630],[6.146],[1.315],[0.538],[2.658],[4.301],[2.058],[5.947],[0.702],[5.713],[0.163]])
        self.upvar = np.array([[0.708],[3.846],[0.172],[3.630],[6.146],[1.315],[0.538],[2.658],[4.301],[2.058],[5.947],[0.702],[5.713],[0.163]])
        self.alphas = np.array([[25],[15],[37],[19],[51],[40],[51],[29],[12],[26],[11],[38],[25],[30]])
        self.betas = np.array([[384],[213],[274],[220],[458],[331],[480],[376],[230],[312],[186],[413],[285],[438]])
        self.varprobs = np.zeros(14)

        for i in range(14):
                self.varprobs[i] = (self.alphas[i] / (self.alphas[i] + self.betas[i] + 1))


    def bid(self, attr):
        """
        returns a random bid, regardless of attribute
        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random
        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """

        a_ix = self.attrs.index(attr)
        

        #confidence interval is based on probability
        vkg = np.zeros(14)
        
        for i in range(0,13):
            vkg[i] = self.mean[i] + 0.98 * self.upvar[i]
        
        choice = np.floor(vkg[a_ix])

        return choice

    def learn(self, info):
        """
        learns from auctions results
        This policy does not learn (need not learn, because it just bids randomly)
        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        #Updating Mean
        for result in info:
            #print("result: ", result)
            #print("aix", a_ix)
            if result['cost_per_click'] == '':
                continue
            attr = result['attr']
            a_ix = self.attrs.index(attr)
            #print(a_ix)
            self.mean[a_ix] = ((((1.0 / self.upvar[a_ix]) * self.mean[a_ix]) + (result['cost_per_click'] * (1.0 / self.var[a_ix]))) / ((1 / self.var[a_ix]) + (1 / self.upvar[a_ix])))
            #Updating Variance
            self.upvar[a_ix] = (1.0 / ((1.0 / self.upvar[a_ix]) + (1.0 / self.var[a_ix])))

            revenue = result['revenue_per_conversion']
            cost = result['cost_per_click']
            #Updating Beta and Alpha
            if result['cost_per_click'] == '' and result['revenue_per_conversion'] * result['num_conversion'] < result['num_click'] * result['cost_per_click']:
                self.betas[a_ix] += 1
            else:
                self.alphas[a_ix] += 1
            
        return True