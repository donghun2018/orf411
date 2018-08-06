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

    


class Policy_ohiustina_LA_5(Policy):
    

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
        self.smaller_bid_space  = np.arange(5, 10, 1)
        self.max_t = max_t
        self.prng = np.random.RandomState(randseed)
        s = np.matrix([[-1.62867438, -10.09666099, -2.78465446, 4.62867438, -1.94707818,-2.81714848],
                       [ 0.67439152,  4.5914827 ,  0.85327314,  -0.67439152,  2.77829298, 1.71436637],
                       [-1.37994894, -0.44307331, -4.30853397, 0.37994894, -0.36496003, -0.37529304],
                       [ 5.95615085, -3.01747743, -0.50161148,  -3.95615085, -0.6561599 , 6.34741058],
                       [ 2.57833508, -10.37614397, -1.41401465,  -1.57833508, -2.14946838, -3.75659137],
                       [ 0.52793778, -0.71639215, 1.38121725,  3.52793778, -0.46753452, -0.10185367],
                       [ -1.91167616, -4.58702131, -1.29406939,  -1.91167616, -2.83503703, -1.97855508],
                       [ 9.59430319, 5.0993157 , -0.64010177,  1.59430319, -2.19806465, -4.14977515],
                       [ 3.00695666, -3.02591238, -2.97920867,  2.00695666, -1.85553641, 1.36934883]])
                
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
        """
        Your bidding algorithm
        Note how this uses internal pseudorandom number generator (self.prng) instead of np.random
        :param attr: attribute tuple. guaranteed to be found in self.attrs
        :return: a value that is found in self.bid_space
        """

        thetaKG = 3 # A tunable parameter
        mu = np.ones(len(self.smaller_bid_space))
        smaller_L_range = range(1,60,20) 

        p = np.zeros((len(self.smaller_bid_space), 6))
        for i in range(len(self.smaller_bid_space)):
            for j in range(6):
                #print(self.theta[j])
                p[i,j]= pwin(self.smaller_bid_space[i], self.theta.loc[:, self.theta.columns[j]].values,  attr)
                mu[i] += self.q_vec[j]*(self.revenue_per_conversion[attr]*self.conversions_per_click[attr] -    self.smaller_bid_space[i])*self.lambda_par[attr]*self.clicks_per_impression[attr]*pwin(self.smaller_bid_space[i], self.theta.loc[:, self.theta.columns[j]].values, attr)   
      

        max_1_array = np.zeros(len(self.smaller_bid_space))
        max_2_array = np.zeros(len(self.smaller_bid_space))
        q_vec_LA = np.ones(6)*1/6
        logq_vec = np.log(q_vec_LA)


        KG_array = np.zeros(len(self.smaller_bid_space))
        OLKG_array = np.zeros(len(self.smaller_bid_space))
        p_bar = np.zeros(len(self.smaller_bid_space))
        n_star = np.zeros(len(self.smaller_bid_space))
        #print('smaller_bid_space',self.smaller_bid_space)
        max_kg = np.ones(len(self.smaller_bid_space))*(-1)



 


        for i in range(len(self.smaller_bid_space)):
            for j in range(6):
                p_bar[i] += self.q_vec[j]*p[i,j]
            #print (p_bar[i])
            #auct = 0  # For auction that wil be randomly generated
            #imp = 0  # For impression that will be randomly generated


            for L in smaller_L_range:
                KG_hat = np.zeros(L)
                for l in range(L):
                    auct = np.random.poisson(lam = self.lambda_par[attr])
                    imp = np.random.binomial(auct, p_bar[i])
                    while(imp > auct):
                        imp = np.random.binomial(auct, p_bar[i])
                    sum_2 = 0
                    rv = np.zeros(6)
                    sh=[]
                    for j in range(6):
                        rv[j] = binom(auct, p_bar[i]).logpmf(imp)
                        sh.append(rv[j] + self.logq_vec[j])
                    sum_2 = logsumexp(sh)
                    #print("sum_2 = ",sum_2)
                    #print(sum_2)  
                    for j in range(6):
                        #print(p[i], rv, self.num_auct[attr],self.impressions[attr])
                        logq_vec[j] += rv[j] - sum_2
                        
                    q_vec_LA = np.exp(logq_vec) 
                   
                    for j in range(6):
                        max_1_array[i] += q_vec_LA[j]*mu[i]
                        #print(max_1_array[i])
                        max_2_array[i] += self.q_vec[j]*mu[i]
                        #print("q_vec_la[",j,']=',q_vec_LA[j])
                        #print("self.q_vec[",j,']=',self.q_vec[j])
                    KG_hat[l] = np.amax(max_1_array) - np.amax(max_2_array)
                    #print(KG_hat[l])
                KG_array[i] = np.sum(KG_hat)/L                
                if(KG_array[i] >= max_kg[i]):
                    max_kg[i] = KG_array[i]
                    n_star[i] = L
            #print("mu[",i,'] = ',mu[i])
            #print('max_kg[',i,'] = ',max_kg[i])
            #print('n_star[',i,']=',n_star[i])
            if((max_kg[i]!=0) & (mu[i] != 0)):
                s = int(round(np.log(abs(mu[i])/(abs(max_kg[i])))))
            else:
                s = 0
            OLKG_array[i] = mu[i] + (168 - self.iter)*max_kg[i] * 3 *10**s
            
            
            




        x = np.argmax(OLKG_array)
        
        self.bid_price[attr] = self.smaller_bid_space[x] 
        
        return self.bid_price[attr]


    def learn(self, info):

        b=0
        p = np.zeros((len(self.smaller_bid_space), 6))
        list_item = [(i,j)
                     for i in range(len(self.smaller_bid_space))
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
                p[i,j] = pwin(self.smaller_bid_space[i], self.theta.loc[:, self.theta.columns[j]].values, attr)

            sh = []
            
            #UPDATING EQUATIONS:
            #ATTRIBUTES
            
        
            #IMPRESSIONS
            #print('self.impressions[',attr,'] = ', self.impressions[attr])
            b =np.where(self.smaller_bid_space == result['your_bid'])
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


