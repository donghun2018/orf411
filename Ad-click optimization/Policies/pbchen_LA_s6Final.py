"""
Patrick Chen
May 14 2018 LA KG Final
"""


import numpy as np
import math as math

from .policy import Policy    # this line is needed


class Policy_pbchen_LA_s6Final(Policy):

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

        self._initialize_state_var()

    def _initialize_state_var(self):
        """
        initial parameters are set up
        """
        self.max_rounds = 168
        stable = 10
        self.n = 1
        self.num_auction = []
        self.rev = []

        self.num_click = np.ones(len(self.attrs))*stable
        self.num_conv = np.ones(len(self.attrs))*stable*.1 # p convert = .1
        self.num_impr = np.ones(len(self.attrs))*stable  * 3 # pclick = .33

        self.n_theta = 100
        self.kg_var = 150

        self.theta = [] 
        self.theta_belief = []

        for i in range(len(self.attrs)):
            self.num_auction.append(120)
            self.rev.append(50)

            # Generate thetas
            const = self.prng.normal(5, 2, size = self.n_theta)
            price_sensitivity = self.prng.normal(1, .2, size = self.n_theta)
            gen_thetas = np.column_stack((const, price_sensitivity))
            self.theta.append(gen_thetas)

            # Uniform initial beliefs about thetas
            initial_theta_belief =  np.ones(self.n_theta)/self.n_theta
            self.theta_belief.append(initial_theta_belief)

    def _calc_presult(self, attr, bid, result):
        a_ix = self.attrs.index(attr)
        params = self.theta[a_ix]
        presult = []

        for theta in params:
            p_win = 1/(1+math.exp(np.sum(theta * np.array([1, -bid]))))
            if result == 0:
                p_iresult = 1 - p_win
            else:
                p_iresult = p_win
            presult.append(p_iresult)
        return presult

    def get_prior(self):
        values = []

        values.append(self.num_auction)
        values.append(self.num_click/self.num_impr)
        values.append(self.num_conv/self.num_click)
        values.append(self.rev)

        best_theta0 = []
        best_thetax = []
        for i in range(len(self.attrs)):
            a_ix = np.argmax(self.theta_belief[i])
            best_theta = self.theta[i][a_ix]
            best_theta0.append(best_theta[0])
            best_thetax.append(best_theta[1])

        best_theta = []
        best_theta.append(np.mean(best_theta0))
        best_theta.append(np.std(best_theta0)) 
        best_theta.append(np.mean(best_thetax))
        best_theta.append(np.std(best_thetax))
        values.append(best_theta) # mean theta0, std theta0, mean thetax, std thetax
        return(values)

    def bid(self, attr):
        """
        finds a bid that is closest to the revenue sample mean of auctions with the given attr

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        a_ix = self.attrs.index(attr)
        exp_profit = []
        

        cache_win = []
        cache_lose = []
        cache_prof = []

        for i in range(len(self.bid_space)):
            # Calculate the expected profit
            bid = self.bid_space[i]
            p_win_theta = self._calc_presult(attr, bid, 1) # each theta belief about winning
            p_lose_theta = self._calc_presult(attr, bid, 0) # each theta belief about losing

            cache_win.append(np.asarray(p_win_theta))
            cache_lose.append(np.asarray(p_lose_theta))

            if bid <= 3:
                cost = bid
            elif bid <= 5.5:
                cost = (bid-3)*.2+3
            else:
                cost = bid - 2

            pred_pwin = p_win_theta * self.theta_belief[a_ix] # weight by belief about theta

            prof_temp = (self.num_auction[a_ix] * (self.num_click[a_ix]/self.num_impr[a_ix]) * 
                (self.rev[a_ix]*(self.num_conv[a_ix]/self.num_click[a_ix]) - cost))
            cache_prof.append(prof_temp)

            profit_from_bid =  prof_temp * sum(pred_pwin)
            exp_profit.append(profit_from_bid)

        list1 = [x*10+2 for x in range(10)]
        list2 = [x*10+7 for x in range(10)]
        bids_spot = list1+list2

        red = len(self.bid_space)

        v_kg = np.zeros(red)
        belief = self.theta_belief[a_ix]
        for result in range(2):
            for bidprime in bids_spot:
                temp = np.zeros(red)
                if result == 0:
                    cache_res = cache_lose
                else:
                    cache_res = cache_win

                for j in range(len(self.bid_space)):
                    temp[j] = sum(cache_res[bidprime] * cache_win[j] * belief) * cache_prof[j]
                    #temp[j] = sum(cache_res[i*10] * cache_win[j*10] * belief) * cache_prof[j]

                v_kg[bidprime] = v_kg[bidprime] + max(temp)
        

        v_kg = (v_kg - max(exp_profit)) * max(168 - self.n,1) * self.kg_var
        for i in range(len(v_kg)):
            if v_kg[i] < 0:
                v_kg[i] = 0

        next = exp_profit + v_kg

        select = [next[i] for i in bids_spot]

        bid = bids_spot[np.argmax(select)] + self.prng.randint(-2, 3)

        # print(np.argmax(exp_profit)/10, bid/10, "|", np.mean(exp_profit), max(exp_profit), min(exp_profit),
        #  "|", max(v_kg), np.mean(v_kg))
        #print(bid)

        # Note we should never bid too low should always bid > 0
        if(bid <= 5):
            bid = self.prng.randint(10, 60)

        return(self.bid_space[bid])
            

    def learn(self, info):
        """
        learns from auctions results

        In this sample policy, it learns by keeping track of sample averages of revenue from auctions of each attribute.
        If 'revenue_per_conversion' is empty, that means I did not get any conversions in those auctions. I ignore them.

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        self.n = self.n+1

        for result in info:
            # Get attribute index
            a_ix = self.attrs.index(result['attr'])

            # Update number of expected auctions
            self.num_auction[a_ix] = ((1-1/self.n)*self.num_auction[a_ix] 
                + 1/self.n*result['num_auct'])

            # Calculate probability of auction result
            p_auc_result = self._calc_presult(result['attr'],
                result['your_bid'], int(result['your_bid'] == result['winning_bid']))

            # Update p(True theta)
            self.theta_belief[a_ix] = p_auc_result * self.theta_belief[a_ix]  
            self.theta_belief[a_ix] = self.theta_belief[a_ix]/sum(self.theta_belief[a_ix])

            # update number of clicks
            self.num_click[a_ix] = self.num_click[a_ix] + result['num_click']
            # update number of impressions
            self.num_impr[a_ix] = self.num_impr[a_ix] + result['num_impression']

            # update number of conversions
            old_num_conv = self.num_conv[a_ix]
            self.num_conv[a_ix] = self.num_conv[a_ix]  + result['num_conversion']

            # Update expected revenue
            if result['revenue_per_conversion'] != '':
                self.rev[a_ix] = ((self.rev[a_ix]*old_num_conv + 
                    result['revenue_per_conversion'] * result['num_conversion'])/
                    self.num_conv[a_ix])

        return True






