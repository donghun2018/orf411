""" Bidding Policy base class Donghun Lee 2018 All policy classes inherit this class. """

import numpy as np
from .policy import Policy    # this line is needed


class Policy_tw5LA(Policy):

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
        
        self.avg_cost_per_click = np.array([[2.001],[2.512],[1.932],[2.428],[2.689],[7.015],[4.283],[4.026],[4.511],[1.282],[4.589],[1.816],[2.801],[2.198]])
        self.var_cost_per_click =np.array([[0.934],[3.816],[0.244],[2.355],[2.287],[2.293],[2.150],[2.489],[2.311],[2.286],[2.435],[2.525],[2.121],[1.856]])
        self.updating_var = np.array([[0.934],[3.816],[0.244],[2.355],[2.287],[2.293],[2.150],[2.489],[2.311],[2.286],[2.435],[2.525],[2.121],[1.856]])

    def bid(self, attr):
        """
        returns a random bid, regardless of attribute
        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random
        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        # print(attr)
        #print(self)
        #print(self.attrs)
        #a_ix = 0
        a_ix = self.attrs.index(attr)
        
        #print(a_ix)

        vkg = np.zeros(14)
        
        for i in range(0,13):
            vkg[i] = self.avg_cost_per_click[i] + 0.8 * self.var_cost_per_click[i]
        
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
            index = self.attrs.index(attr)
            #print(a_ix)
            self.avg_cost_per_click[index] = ( ((1.0/self.updating_var[index])*self.avg_cost_per_click[index]) + ((1.0 / self.var_cost_per_click[index])*result['cost_per_click'])) / ((1 / self.var_cost_per_click[index]) + (1 / self.updating_var[index]))
            self.updating_var[index] = (1.0 / ((1.0 / self.updating_var[index]) + (1.0 / self.var_cost_per_click[index])))
            
        return True