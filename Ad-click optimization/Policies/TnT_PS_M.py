"""
Todd meets Trevor 2K18

Upper Confidence Bounding

Bids sample average of observed revenue
"""


import numpy as np

from .policy import Policy    # this line is needed


class Policy_TnT_PS_M(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12346):
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
        
        self.bid_space_small = []
        for i in range(len(self.bid_space)):
            if i % 4 == 0:
                self.bid_space_small.append(self.bid_space[i])
        self.mu_0 = {}
        self.mu_cnt = []
        self.sigma = []
        self.uncertainty = []
        self.N_x = [1]*len(self.bid_space_small) # N in the UCB policy
        self.iter = 0
        profit = [0]*len(self.bid_space_small)
        for i in range(len(self.bid_space_small)):
            if self.bid_space_small[i] < 8.6:
                profit[i] = 5*np.exp(-(8.6-self.bid_space_small[i]))
            else:
                profit[i] = 5*np.exp(self.bid_space_small[i]-8.6)
        
        for attr in all_attrs:
            #init_val = self.prng.choice(self.bid_space)
            self.mu_cnt.append(0)
            self.uncertainty.append(0)
            self.mu_0[attr] = profit.copy()

    def _update_estimates(self, attr, profit, bid):
        """
        iterative averaging of samples
        :param attr: attribute
        :param x: sample observed
        :return: None
        """
        a_ix = self.attrs.index(attr)
        bid_ix = self.bid_space_small.index(bid) #update mu_0 for this particular bid only
        mu_0 = self.mu_0[attr][bid_ix]
        n = self.mu_cnt[a_ix] + 1
        mu2 = 1/n * profit + (n-1)/n * mu_0
        self.mu_0[attr][bid_ix] = mu2
        self.mu_cnt[a_ix] += 1
        
    def UCB(self, attr, nn, m):
        # number of available choices
        kk = len(self.bid_space_small)
        # this is the theta
        theta = 3.3333
        if self.iter < kk:
            return self.iter
        else:
            nu = [0]*kk
            for k in range(kk):
                nu[k] = self.mu_0[attr][k] + np.sqrt(theta/self.N_x[k])
            max_choice = np.argmax(nu)
        return max_choice
       
        
    def bid(self, attr):
        """
        finds a bid that is closest to the revenue sample mean of auctions with the given attr
        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
    
        a_ix = self.attrs.index(attr)
        nn = 100
        m = 1
        upper_bid = self.bid_space_small[self.UCB(attr, nn, m)]
        self.iter += 1
        
        return upper_bid
    
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
            if result['cost_per_click'] == '':
                continue
            attr = result['attr']
            if result['revenue_per_conversion'] == '':
                profit = -result['num_click'] * result['cost_per_click']
            else:
                profit = (result['revenue_per_conversion'] * result['num_conversion']) - (result['num_click'] * result['cost_per_click'])
            self._update_estimates(attr, profit, result['your_bid'])

        return True
