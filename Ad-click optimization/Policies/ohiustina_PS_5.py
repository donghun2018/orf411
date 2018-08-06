import numpy as np
import pandas as pd
from .policy import Policy
from scipy.special import comb
from scipy.stats import binom
from scipy.misc import logsumexp


def binarize(n, k):
    y = [0] * k
    y[n] = 1
    return y

def expand_attr(attr):
    bid, (gender, age) = attr
    s = [1, bid, gender] + binarize(age, 7)
    return s[slice(0,9)]


def ufct(x, theta, attr):
  #x is a vector containing our bid and the user's attributes 
    attrs = (x, attr)
    long_attr = expand_attr(attrs)
    u = np.dot(long_attr, theta)        
    return u


def pwin(x, theta, attr):
    p_win = 1 / ( 1 + np.exp(- ufct(x, theta, attr))) 
    return(p_win)


def print_self(self, attr):
    print("q_vec = ",self.q_vec)
    print("self.clicks_per_impression[attr]",self.clicks_per_impression[attr])
    print("self.lambda_par[attr]",self.lambda_par[attr])
    print("self.conversions[",attr," ] = ", self.conversions[attr])
    print("self.conversions_per_click[",attr," ] = ", self.conversions_per_click[attr])
    return True

    


class Policy_ohiustina_PS_5(Policy):
    

    def __init__(self, all_attrs, possible_bids=np.linspace(0,10,100), max_t=10, randseed=67890):
        """
        Your bidding policy class initializer
        Note that the first line must be that super().__init__ ...
        Please use self.prng. instead of np.random. if you want to give randomness to your policy
        :param all_attrs: list of all possible attributes
        :param possible_bids: list of all allowed bids
        :param max_t: maximum number of auction 'iter', or iteration timesteps
        :param randseed: random number seed.
        """
        super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)
    
        
        self.attrs = all_attrs
        self.bid_space = possible_bids
        self.max_t = max_t
        self.prng = np.random.RandomState(randseed)
        s = np.matrix([[-4.62867438, -0.09666099, -2.78465446, -4.62867438, -1.94707818,-2.81714848],
                       [ 0.67439152,  0.5914827 ,  0.85327314,  0.67439152,  0.77829298, 0.71436637],
                       [-0.37994894, -0.44307331, -0.30853397, -0.37994894, -0.36496003, -0.37529304],
                       [ 1.95615085, -1.01747743, -0.50161148,  1.95615085, -0.6561599 , 0.34741058],
                       [ 1.57833508, -3.37614397, -1.41401465,  1.57833508, -2.14946838, -0.75659137],
                       [ 0.52793778, -0.71639215, -0.38121725,  0.52793778, -0.46753452, -0.10185367],
                       [ 1.91167616, -4.58702131, -1.29406939,  1.91167616, -2.83503703, -0.97855508],
                       [ 1.59430319, -2.0993157 , -0.64010177,  1.59430319, -1.19806465, -0.14977515],
                       [ 2.00695666, -3.02591238, -0.97920867,  2.00695666, -1.85553641, -0.36934883]])
                
        self.theta = pd.DataFrame(s)
        self.bid_price = {}
        self.revenue_per_conversion = {}
        self.clicks = {} 
        self.impressions = {}
        self.conversions = {}
        self.clicks_per_impression = {}
        self.conversions_per_click = {}
        self.lambda_par = {}
        self.iter = 0
        self.num_auct={}
        
 
        
        for attr in all_attrs:
            self.bid_price[attr] = self.prng.choice(self.bid_space)
            self.revenue_per_conversion[attr] = 100
            self.clicks[attr] = 1
            self.impressions[attr] = 1
            self.clicks_per_impression[attr] = 0.8
            self.conversions[attr] = 1
            self.conversions_per_click[attr] = 0.5     
            self.lambda_par[attr] = 100
            self.q_vec = np.zeros(6)
            self.logq_vec = np.zeros(6)
            self.N = np.ones(len(self.bid_space))
            self.q_vec[:] = 1/6
            self.logq_vec[:] = np.log(self.q_vec)
            self.num_auct[attr] = 0

    def bid(self, attr):
                
        list_item = [(i,j)
                     for i in range(len(self.bid_space))
                     for j in range(6)]
        
        thetaUCB = 4
        UCB_array = np.zeros(len(self.bid_space))
        mu = np.zeros(len(self.bid_space))
        p = np.zeros((len(self.bid_space),6))
        if self.iter == 0:
            self.bid_price[attr] = 9.9 # Initialize the bid to 7
            return self.bid_price[attr]
            exit
        
        for i in range(len(self.bid_space)):
            for j in range(6):  
                p[i,j] = pwin(self.bid_space[i], self.theta.loc[:, self.theta.columns[j]].values, attr) 
                rev = self.revenue_per_conversion[attr]*self.conversions_per_click[attr] -  min(self.bid_space[i],6) 
                #print('rev', rev)
                mu[i] += self.q_vec[j]*(rev * self.lambda_par[attr]*self.clicks_per_impression[attr]* p[i,j])
                #print(mu[i],'mu[i]')
      
        
        for i in range(len(self.bid_space)):
            UCB_array[i] = mu[i] + thetaUCB * np.sqrt(2*np.log(self.iter)/self.N[i])

        x = np.argmax(UCB_array)
        self.N[x] += 1
        self.bid_price[attr] = self.bid_space[x] 
        return self.bid_price[attr]
        

    def learn(self, info):

        b=0
        p = np.zeros((len(self.bid_space),6))
        list_item = [(i,j)
                     for i in range(len(self.bid_space))
                     for j in range(6)]
        
        for result in info: 

            self.iter = result['iter']
            attr = result['attr']
            #print("BEFORE LEARNING iteration", self.iter)
            #print_self(self,attr)
            self.num_auct[attr] += result['num_auct']
          #  if result['num_auct'] < result['num_impression']:
                #print("HAAAAA",result['num_auct'], result['num_impression'])
           #     self.num_auct[attr] += result['num_impression']-result['num_auct']
            self.impressions[attr] += result['num_impression']
     
            for i, j  in list_item:
                p[i,j] = pwin(self.bid_space[i], self.theta.loc[:, self.theta.columns[j]].values, attr)

            sh = []
            
            #UPDATING EQUATIONS:
            #ATTRIBUTES
            
        
            #IMPRESSIONS
            #print('self.impressions[',attr,'] = ', self.impressions[attr])
            b = np.int32(10 * (result['your_bid']))
            rv = np.zeros(6)
            for j in range(6):
                rv[j] = binom(result['num_auct'], p[b,j]).logpmf(result['num_impression'])
                #print('rv[',j,'] = ',rv[j])
                sh.append(rv[j] + self.logq_vec[j])
            sum_2 = logsumexp(sh)


            #Q-VECTOR

            for j in range(6):
                 self.logq_vec[j] += rv[j] - sum_2
            self.q_vec = np.exp(self.logq_vec) 
            if any(np.isnan(self.q_vec)):
                raise Exception()

            #LAMBDA
            self.lambda_par[attr] = (result['iter'] * self.lambda_par[attr] + result['num_auct'])/(result['iter']+1)
            if result['num_click'] > 0:
                    old_revenue = self.revenue_per_conversion[attr]*self.conversions[attr]
                    self.clicks[attr] += result['num_click']

                    self.clicks_per_impression[attr] =  self.clicks[attr]/self.impressions[attr]
                    if result['num_conversion'] > 0:
                        add_revenue = result['num_conversion']*result['revenue_per_conversion']
                        self.conversions[attr] += result['num_conversion']      
                        self.conversions_per_click[attr] = self.conversions[attr]/self.clicks[attr]
                    else:
                        add_revenue = 0

                    self.revenue_per_conversion[attr] = (old_revenue + add_revenue)/self.conversions[attr]
            #print("AFTER LEARNING iteration", self.iter)
            #print_self(self,attr)
        return True

   