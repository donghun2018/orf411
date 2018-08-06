"""
Thompson Sampling Policy (TSP)
William Kang & Tim Thong 2019
Given sufficient information, bid price is based on Thompson Sampling Policy
Else, with certain percentage, bid random, else, bid 0
"""

import textwrap
import numpy as np
import math
import random
from .policy import Policy    # this line is needed
import os,sys,inspect
import os.path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import auction
import sim_lib as sl
from auction import Auction
from simulator import Simulator

class Policy_Weebs_LA_MetropolisHastings(Policy):

    def __init__(self, all_attrs, possible_bids=list([v / 10 for v in range(100)]), max_t=10, randseed=random.randint(1,10000)):
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
        # Initialize priors
        self.iterations = max_t
        self.sim = Simulator()
        self._initialize_priors(max_t)

    def _initialize_priors(self,iterations):

        #  Tunable Parameters
        global num_of_ad_slots
        global ad_slot_click_prob_adjuster
        max_prior_iterations = round(float(3*np.sqrt(iterations)))
        random_players = len(os.listdir("Policies"))-5
        num_of_ad_slots = self.sim.num_of_ad_slots

        file = open(os.path.dirname(__file__) + '/../simulator.py')
        lines = file.readlines()
        for line in lines:
            if 'click_prob_adjuster' in line and 'self.num_of_ad_slots' in line:
                code_extractor = line
        file.close()

        #  ------------------

        num_bids = len(range(100))
        """
            Start with empty priors - learn along the way model
        """
        self.iter = 0
        self.theta_array = [None] * len(self.attrs)
        self.lambda_array = [None] * len(self.attrs)
        self.avg_revenue_array = [None] * len(self.attrs)
        self.prob_conversion_array = [None] * len(self.attrs)

        # List of Revenues and Clicks for Each Attribute
        self.profit = np.zeros((len(self.attrs),num_bids))
        self.profit_count = np.ones((len(self.attrs), num_bids))*max_prior_iterations/3

        param, attrs = Auction.read_init_xlsx("")
        auction = Auction(param, attrs)
        gen = auction.generate_sample()
        random_bid_list = list([v / 10 for v in range(100)])
        count = 0

        if random_players+1 < num_of_ad_slots:
            num_of_ad_slots = random_players + 1

        for a in gen:
            if count >= len(self.attrs): break

            self.theta_array[count] = a['theta']
            self.lambda_array[count] = a['lambda']
            self.avg_revenue_array[count] = a['avg_revenue']
            self.prob_conversion_array[count] = a['prob_conversion']
            count += 1
        for i in range(len(self.attrs)):
            for j in range(num_bids):
                for k in range(max_prior_iterations):
                    num_profit = 0
                    code_extractor = textwrap.dedent(code_extractor)
                    code_extractor2 = code_extractor.replace('self.','')
                    exec(code_extractor2, globals())
                    ad_slot_click_prob_adjuster = [p / sum(click_prob_adjuster) for p in click_prob_adjuster]
                    random_players_array = [None] * random_players
                    for ix in range(random_players):
                        random_players_array[ix] = float(random.choice(random_bid_list))
                    random_players_array.append(float(j/10))
                    theta = {'a': self.theta_array[i],
                             'bid': 0,
                             '0': 0,
                             'max_click_prob': 0.5}
                    p_click = sl.get_click_prob(theta, max(random_players_array))
                    num_auct = self.prng.poisson(self.lambda_array[i])
                    num_clicks = self.prng.binomial(num_auct, p_click)
                    reverse_sorted_bids, sorted_pIx = sl.top_K_max(random_players_array, num_of_ad_slots, self.prng)
                    actual_clicks = 0
                    sorted_unique_bids = sorted(list(set(random_players_array)))
                    actual_cost = []
                    for ix in range(num_clicks):
                        winner_ix = int(self.prng.choice(sorted_pIx, p=ad_slot_click_prob_adjuster))
                        if random_players_array[winner_ix] == j/10:
                            actual_clicks += 1
                            actual_cost.append(sl._compute_actual_second_price_cost(j/10, sorted_unique_bids))

                    array_revenue = Auction.get_revenue_sample(self.avg_revenue_array[i], self.prng, size=actual_clicks)
                    for ix in range(actual_clicks):
                        num_profit += self.prob_conversion_array[i] * array_revenue[ix] - actual_cost[ix]

                    self.profit[i][j] = self.profit[i][j] * k / (k + 1) + num_profit / (k + 1)


    def _update_prior_estimate(self, attr, revenue, numclick, last_bid, second_price, winning_bid, winning_bid_avg):
        ind = self.attrs.index(attr)
        num_bids = len(range(100))
        lower_bound = max(2*winning_bid_avg-winning_bid, 0)
        lower_bound_index = int(math.ceil(lower_bound * 10))
        max_bound_index = int(math.ceil(winning_bid * 10))
        if last_bid > lower_bound:
            theta = {'a': self.theta_array[ind],
                     'bid': 0,
                     '0': 0,
                     'max_click_prob': 0.5}
            p_click = sl.get_click_prob(theta, winning_bid)
            num_auct = self.prng.poisson(self.lambda_array[ind])
            new_random_players = self.sim.num_of_ad_slots - 4
            new_random_players_array = [None] * new_random_players
            new_random_bid_list = np.arange(lower_bound+0.1,winning_bid-0.1,0.1)
            new_profit = np.zeros(num_bids)
            for index in range(lower_bound_index+1,max_bound_index-1):
                for trial in range(50):
                    for ix in range(new_random_players):
                        new_random_players_array[ix] = float(random.choice(new_random_bid_list))
                    new_random_players_array.append(index/10)
                    new_random_players_array.append(lower_bound)
                    new_random_players_array.append(winning_bid)
                    new_random_players_array.append(second_price)
                    theta = {'a': self.theta_array[ind],
                             'bid': 0,
                             '0': 0,
                             'max_click_prob': 0.5}
                    p_click = sl.get_click_prob(theta, max(new_random_players_array))
                    num_auct = self.prng.poisson(self.lambda_array[ind])
                    num_clicks = self.prng.binomial(num_auct, p_click)
                    reverse_sorted_bids, sorted_pIx = sl.top_K_max(new_random_players_array, self.sim.num_of_ad_slots, self.prng)
                    actual_clicks = 0
                    sorted_unique_bids = sorted(list(set(new_random_players_array)))
                    actual_cost = []
                    for ix in range(num_clicks):
                        winner_ix = int(self.prng.choice(sorted_pIx, p=ad_slot_click_prob_adjuster))
                        if new_random_players_array[winner_ix] == index / 10:
                            actual_clicks += 1
                            actual_cost.append(sl._compute_actual_second_price_cost(index / 10, sorted_unique_bids))
                    num_profit = 0
                    array_revenue = Auction.get_revenue_sample(self.avg_revenue_array[ind], self.prng, size=actual_clicks)
                    for ix in range(actual_clicks):
                        num_profit += self.prob_conversion_array[ind] * array_revenue[ix] - actual_cost[ix]

                    new_profit[index] = new_profit[index] * trial / (trial + 1) + num_profit / (trial + 1)

            self.profit[ind][lower_bound_index+1:max_bound_index-1] = self.profit[ind][lower_bound_index+1:max_bound_index-1]\
                                * self.profit_count[ind][lower_bound_index+1:max_bound_index-1] / (self.profit_count[ind][lower_bound_index+1:max_bound_index-1] + 1) \
                              + new_profit[lower_bound_index+1:max_bound_index-1] / (self.profit_count[ind][lower_bound_index+1:max_bound_index-1] + 1)
            self.profit_count[ind][lower_bound_index+1:max_bound_index-1] += 1
        else:
            k = self.profit_count[ind][0:(lower_bound_index)]
            self.profit[ind][0:(lower_bound_index)] = self.profit[ind][0:(lower_bound_index)]* k / (k+1)
            self.profit_count[ind][0:(lower_bound_index)] += 1

        for index in range(max_bound_index,100):
            theta = {'a': self.theta_array[ind],
                     'bid': 0,
                     '0': 0,
                     'max_click_prob': 0.5}
            p_click = sl.get_click_prob(theta, index/10)
            num_auct = self.prng.poisson(self.lambda_array[ind])
            num_clicks = self.prng.binomial(num_auct, p_click)
            actual_clicks = int(math.ceil(num_clicks * ad_slot_click_prob_adjuster[0]))
            actual_cost = winning_bid
            array_revenue = Auction.get_revenue_sample(self.avg_revenue_array[ind], self.prng, size=actual_clicks)
            num_profit = 0
            for ix in range(actual_clicks):
                num_profit += self.prob_conversion_array[ind] * array_revenue[ix] - actual_cost

            self.profit[ind][index] = self.profit[ind][index] * self.profit_count[ind][index]/ (self.profit_count[ind][index]+1) \
            + num_profit /(self.profit_count[ind][index]+1)

        self.profit_count[ind][max_bound_index:100] += 1

        j = int(last_bid*10)
        num_profit = revenue - second_price * numclick
        self.profit[ind][j] = self.profit[ind][j] * self.profit_count[ind][j]/(self.profit_count[ind][j]+1) \
                              + num_profit/(self.profit_count[ind][j]+1)
        self.profit_count[ind][j] += 1

    def bid(self, attr):
        ind = self.attrs.index(attr)

        random_bid_list = list([v / 10 for v in range(100)])
        explore_bid = float(random.choice(random_bid_list))
        exploit_bid = np.argmax(self.profit[ind][:])/10
        x = np.random.uniform(0, 1, 1)
        if exploit_bid < explore_bid:
            if 1 / ((1 + self.iter) ** (exploit_bid - explore_bid)) < x:
                return explore_bid
            else:
                return exploit_bid
        else:
            return exploit_bid

    def learn(self, info):
        """
        learns from auctions results
        Calculates revenue and click number based on the auction results and updates our priors
        :return: does not matter
        """
        self.iter += 1
        for result in info:
            attr = result['attr']
            ind = self.attrs.index(attr)
            # Revenue Calculation

            #  exploration with decreasing probability
            if result['revenue_per_conversion'] == '':
                money = 0
            else:
                money = result['revenue_per_conversion'] * result['num_conversion']

            bid_number = result['your_bid']
            if result['cost_per_click'] == '':
                second_price_bid = 0
            else:
                second_price_bid = result['cost_per_click']
            # Number of Clicks
            curr_click = result['num_click']
            winning_bid = result['winning_bid']
            winning_bid_avg = result['winning_bid_avg']

                # Update priors
            self._update_prior_estimate(attr,money, curr_click, bid_number, second_price_bid, winning_bid, winning_bid_avg)

        return True
