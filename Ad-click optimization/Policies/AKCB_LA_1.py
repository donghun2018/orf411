# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:44:57 2018

@author: qwert
"""

"""
Bidding Policy base class


All policy classes inherit this class.

"""
from scipy.stats import norm
import numpy as np
from numpy.linalg import inv
from .policy import Policy  

class Policy_AKCB_LA_1(Policy):

    def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12345):
        """
        initializes policy base class.

        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)
        self._initialize_beta()
        self._initialize_estimates()
        
        
    def _initialize_beta(self):
        self.beta = []
        self.betaW = .5
        
        for i in range(len(self.attrs)):
            beta_row = []
            for j in range(len(self.bid_space)):
                beta_row.append(0)
            self.beta.append(beta_row)
            
            
    def _initialize_estimates(self):
        self.mu = []
        self.n = []
        self.impressions = []
        self.pClick = []
        self.cPer = []
        self.rPer = []
        self.pCon = []
        self.nIX = []
        self.avg = []
        self.updates = []
        for i in range(len(self.attrs)):
            mu_row = []
            num_row = []
            for j in range(len(self.bid_space)):
                #init_val = self.prng.choice(self.bid_space)
                init_val = 0
                mu_row.append(init_val)
            self.impressions.append(0)   
            self.mu.append(mu_row)
            self.pClick.append(0)
            self.cPer.append(0)
            self.rPer.append(0)
            self.pCon.append(0)
            self.nIX.append(0)
            self.updates.append(0)
            self.n.append(0)
            self.avg.append(1)
            
            
    
    def _get_kgs(self, attr):
        ix = self.attrs.index(attr)
        kg = []
        n = []
        
        for i in range(len(self.bid_space)):
            kg.append(0)
            n.append(0)
        for i in range(len(self.bid_space)):
            muN = self.mu[ix][i]
            betaN = self.beta[ix][i]
            betaN1 = betaN
            kgn = -8000000000000000
            
            while(kg[i] > kgn):
                betaN = betaN1
                betaN1 = betaN + self.betaW
                if(betaN == 0):
                    sig = (1/betaN1)
                else:
                    sig = (1/(betaN)) - (1/betaN1)
                maxn = -80000000
                for j in range(len(self.bid_space)):
                    if((i != j) and (self.mu[ix][j] > maxn)):
                        maxn = self.mu[ix][j]   
                zeta = -(np.abs((muN - maxn)/sig))
                fz = zeta*norm.cdf(zeta) + norm.pdf(zeta)
                kgn = kg[i]
                kg[i] = (kgn*n[i] + fz*sig)/(n[i] + 1)
                n[i] = n[i] + 1
                
            if(kg[i] != 0 and self.mu[ix][i] != 0):   
                self.avg[ix] = (self.avg[ix]*self.n[ix] + np.abs(kg[i]/self.mu[ix][i]))/(self.n[ix] + 1)
                self.n[ix] = self.n[ix] + 1
            
            kg[i] = kg[i]/self.avg[ix] + self.mu[ix][i]
        
        return kg
            
            
            
    def bid(self, attr):
        """
        returns a random bid, regardless of attribute

        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random

        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """
        
        v= self._get_kgs(attr)
        imax = 1
        ival = -80000000
        n = 0
        for i in range(len(v)):
            if(i != 0):
                if(v[i] > ival):
                    imax = i
                    ival = v[i]
                    n = n+ 1
        
        ix = self.attrs.index(attr)
        self.updates[ix] = imax
        
        return self.bid_space[imax]
            
        

    def learn(self, info):
        """
        learns from auctions results

        This policy does not learn (need not learn, because it just bids randomly)

        :param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
                     Follows the same format as output_policy_info_?????.xlsx
        :return: does not matter
        """
        
        for result in info:
            ix = self.attrs.index(result['attr'])
            imp = 0
            if(result['num_impression'] != ''):
                imp = result['num_impression']
            num_conv = 0
            if(result['num_conversion'] != ''):
                num_conv = result['num_conversion']
            num_click = 0
            if(result['num_click'] != ''):
                num_click = result['num_click']
            if(imp != 0):
                self.pClick[ix] = (self.pClick[ix]*self.impressions[ix] + num_click)/(self.impressions[ix] +imp)
                self.impressions[ix] = self.impressions[ix] + imp
                self.pCon[ix] = (self.impressions[ix]*self.pClick[ix]*self.pCon[ix] + num_conv) / (self.impressions[ix]*self.pClick[ix])
            cost_per = 0
            if(result['cost_per_click'] != ''):
                cost_per = result['cost_per_click']
                self.nIX[ix] = self.nIX[ix] + 1
                self.cPer[ix] =  (self.cPer[ix]*(self.nIX[ix] - 1) + cost_per) / (self.nIX[ix])
            rev_p = 0
            if(result['revenue_per_conversion'] != ''):
                rev_p = result['revenue_per_conversion']
                self.rPer[ix] = (self.rPer[ix]*self.nIX[ix] + rev_p) / (self.nIX[ix] + 1)
                
            
            wN = num_click*cost_per - rev_p*num_conv
            wAvg = self.impressions[ix]*self.pClick[ix]*self.cPer[ix] - self.rPer[ix]*self.impressions[ix]*self.pClick[ix]*self.pCon[ix]
            wN = wN - wAvg
            BN = self.beta[ix][self.updates[ix]]
            self.beta[ix][self.updates[ix]] = self.beta[ix][self.updates[ix]] + self.betaW
            muN = (BN*self.mu[ix][self.updates[ix]] + self.betaW * wN) / (BN + self.betaW)
            self.mu[ix][self.updates[ix]] = muN
            
                
                
                
        return True

