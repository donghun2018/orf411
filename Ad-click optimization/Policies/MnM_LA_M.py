"""
Online KG policy with tunable parameters

Midlyn Chen & Matthew Yi 
"""

import numpy as np
import math
from scipy.stats import norm
from .policy import Policy

class Policy_MnM_LA_M(Policy):

	def __init__(self, all_attrs, possible_bids=list(range(10)), max_t=10, randseed=12345):
		"""
		initializes policy base class.

		:param all_attrs: list of all possible attributes
		:param possible_bids: list of all allowed bids
		:param max_t: maximum number of auction 'iter', or iteration timesteps
		:param randseed: random number seed.
		"""
		super().__init__(all_attrs, possible_bids, max_t, randseed=randseed)

		self._initialize_vars()

	def _initialize_vars(self):
		"""
		initial values are set up.

		I chose to initialize with a randomly chosen bid value
		:return:
		"""
		self.num_runs = 168
		self.rho = .85
 
		self.bid_price = {}
		self.iter = {}
		self.var_mu = {}
		self.var_w = {}
		self.covM = {}
		self.mu_0 = {}
		for attr in self.attrs:
			self.bid_price[attr] = self.prng.choice(self.bid_space)
			self.iter[attr] = 1
			self.var_mu[attr] = 100**2
			self.var_w[attr] = 200**2
			temp = [0]*len(self.bid_space)
			for i in range(len(self.bid_space)):
				if i > 60 and i < 77:
					temp[i] = 49
				else:
					temp[i] = 10
			self.mu_0[attr] = temp

			M = len(self.mu_0[attr])
			x = self.bid_space
			covM = [[self.var_mu[attr] for i in range(M)] for j in range(M)]

			for i in range(M):
				for j in range(i, M):
					if i != j:
						covM[i][j] = self.var_mu[attr]*np.exp(-self.rho*np.absolute(x[i]-x[j]))
						covM[j][i] = covM[i][j]

			self.covM[attr] = covM

	def bid(self, attr):
		"""
		:param attr: attribute tuple. guaranteed to be found in self.attrs
		:return: a value that is found in self.bid_space
		"""
		return self.bid_price[attr]

	def learn(self, info):
		"""
		Runs kgcb policy for each observed attribute

		:param info: list of results. Single result is an aggregate outcome for all auctions with a particular attr.
					Follows the same format as output_policy_info_?????.xlsx
		:return: does not matter
		"""

		for result in info:
			attr = result['attr']
			self.iter[attr] = self.iter[attr] + 1
			if result['revenue_per_conversion'] == '':
				continue
			self.iter[attr] = self.iter[attr] + 1
			if result['revenue_per_conversion'] == '':
				profit = -result['num_click'] * result['cost_per_click']
			else:
				profit = (result['revenue_per_conversion'] * result['num_conversion']) - (result['num_click'] * result['cost_per_click'])
			our_bid = int(result['your_bid']*10)

			self._kgcb_update(attr, profit, our_bid)

		return True

	def _kgcb_update(self, attr, profit, our_bid):
		x = self.bid_space
		mu_prev = self.mu_0[attr]
		cov_prev = self.covM[attr]

		ones = [0]*len(x)
		ones[our_bid] = 1

		# update mu_0
		mu_temp = (profit-mu_prev[our_bid])/(self.var_w[attr]+cov_prev[our_bid][our_bid])*np.matmul(cov_prev, ones)
		self.mu_0[attr] = np.add(mu_prev, mu_temp)

		# update covM
		cov_temp = np.multiply(np.matmul(np.matmul(cov_prev, ones), np.transpose(ones)), cov_prev)/((self.var_w[attr]+cov_prev[our_bid][our_bid]))
		self.covM[attr] = np.subtract(cov_prev, cov_temp)

		beta_w = np.multiply(1/self.var_w[attr], [1]*len(x))

		mu_est = []
		for i in range(3):
			mu_est.append(self.kgcb(attr, profit, beta_w, self.iter[attr]))

		mu_est_avg = []
		for i in range(len(x)):
			temp = 0
			for j in range(3):
				temp += mu_est[j][i]
			mu_est_avg.append(temp)

		best_val = max(mu_est_avg)
		best_idx = [i for i, j in enumerate(mu_est_avg) if j==best_val]

		self.bid_price[attr] = x[best_idx[0]]

	# Knowledge Gradient with Correlated Beliefs 

	# notation for the following:
	# K is the number of alternatives.
	# M is the number of time-steps
	# K x M stands for a matrix with K rows and M columns

	# This function takes in
	# mu:     true values for the mean (K x 1)
	# mu_0:   prior for the mean (K x 1)
	# beta_w: measurement precision (1/lambda(x)) (K x 1)
	# cov_m:   initial covariance matrix (K,K)
	# m:      how many measurements will be made (scalar)

	# And returns
	# mu_est:     Final estimates for the means (K x 1)
	# oc:         Opportunity cost at each iteration (1 x M)
	# choices:    Alternatives picked at each iteration (1 x M)
	# mu_est_all:  Estimates at each iteration (K x M)
	def kgcb(self, attr, profit, beta_w, k):
		mu_0 = self.mu_0[attr]
		cov_m = self.covM[attr]
		num_alt = len(mu_0) # number of available choices
		mu_est = mu_0.copy()
		choices = []
		mu_est_all = []
		# py is the KG for alternatives
		py = []
		for i in range(num_alt):
			a = mu_est.copy()
			b = np.divide(cov_m[i], np.sqrt(1/beta_w[i]+cov_m[i][i]))
			kg = EmaxAffine(a,b)

			# For online learning 
			kg = mu_est[i] + (self.num_runs-k) * kg

			py.append(kg)

		x = np.argmax(py)

		# max_value is the best estimated value of the KG
		# x is the argument that produces max_value

		# observe the outcome of the decision
		# w_k=mu_k+Z*SigmaW_k where SigmaW is standard deviation of the
		# error for each observation
		w_k = profit

		# updating equations for Normal-Normal model with covariance
		addscalar = (w_k - mu_est[x])/(1/beta_w[x] + cov_m[x][x])
		# cov_m_x is the x-th column of the covariance matrix cov_m
		cov_m_x = np.array([row[x] for row in cov_m])
		mu_est = np.add(mu_est, np.multiply(addscalar, cov_m_x))
		cov_m = np.subtract(cov_m, np.divide(np.outer(cov_m_x, cov_m_x), 1/beta_w[x] + cov_m[x][x]))

		# pick the best one to compare OC
		max_choice = np.argmax(mu_est)

		# update the choice vector
		choices.append(x)
		# update the matrix of estimate
		mu_est_all.append(mu_est)
		return mu_est


# Calculate the KG value defined by
# E[max_x a_x + b_x Z]-max_x a_x, where Z is a standard
# normal random variable and a,b are 1xM input vectors.
def EmaxAffine(a, b):
	a, b = AffineBreakpointsPrep(a, b)

	c, keep = AffineBreakpoints(a, b)
	keep = [int(keep[i]) for i in range(len(keep))]
	a = a[keep]
	b = b[keep]
	c = np.insert(c[np.add(keep, 1)], 0, 0)
	M = len(keep)

	logbdiff = np.log(np.diff(b))

	if M == 1:
		logy = np.log(a)
	elif M >= 2:
		logy = LogSumExp(np.add(logbdiff, LogEI(-np.absolute(c[1:M]))))

	y = np.exp(logy)
	return y


# Prepares vectors for passing to AffineEmaxBreakpoints, changing their
# order and removing elements with duplicate slope.
def AffineBreakpointsPrep(a, b):
	a = np.array(a)
	b = np.array(b)

	order = np.lexsort((a, b))
	a = a[order]
	b = b[order]

	keep = [i for i in range(len(b) - 1) if b[i] < b[i + 1]]
	keep.append(len(b) - 1)

	a = a[keep]
	b = b[keep]
	return a, b


# Inputs are two M-vectors, a and b.
# Requires that the b vector is sorted in increasing order.
# Also requires that the elements of b all be unique.
# This function is used in AffineEmax, and the preparation of generic
# vectors a and b to satisfy the input requirements of this function are
# shown there.

# The output is an (M+1)-vector c and a vector A ("A" is for accept).  Think of
# A as a set which is a subset of {1,...,M}.  This output has the property
# that, for any i in {1,...,M} and any real number z,
#   i \in argmax_j a_j + b_j z
# iff
#   i \in A and z \in [c(j+1),c(i+1)],
#   where j = sup {0,1,...,i-1} \cap A.
def AffineBreakpoints(a, b):

	M = len(a)
	c = np.array([None] * (M + 1))
	A = np.array([None] * M)

	c[0] = -float("inf")
	c[1] = float("inf")
	A[0] = 0
	Alen = 0

	for i in range(M - 1):
		c[i + 2] = float("inf")
		while True:
			j = A[Alen]  # jindex = Alen
			c[1 + j] = (a[j] - a[i + 1]) / (b[i + 1] - b[j])
			if Alen > 0 and c[1 + j] < c[1 + A[Alen - 1]]:
				Alen -= 1  # Remove last element j
                # continue in while loop
			else:
				break  # quit while loop
		A[Alen+1] = i + 1
		Alen += 1
	A = A[0:Alen+1]
	return c, A


# Returns the log of E[(s+Z)^+], where s is a constant and Z is a standard
# normal random variable.  For large negative arguments E[(s+Z)^+] function
# is close to 0.  For large positive arguments, the function is close to the
# argument.  For s large enough, s>-10, we use the formula
# E[(s+Z)^+] = s*normcdf(s) + normpdf(s).  For smaller s we use an asymptotic
# approximation based on Mill's ratio.  EI stands for "expected improvement",
# since E[(s+Z)^+] would be the log of the expected improvement by measuring
# an alternative with excess predictive mean s over the best other measured
# alternative, and predictive variance 0.
def LogEI(s):
# Use the asymptotic approximation for these large negative s.  The
# approximation is derived via:
#   s*normcdf(s) + normpdf(s) = normpdf(s)*[1-|s|normcdf(-|s|)/normpdf(s)]
# and noting that normcdf(-|s|)/normpdf(s) is the Mill's ratio at |s|, which is
# asymptotically approximated by |s|/(s^2+1) [Gordon 1941, also documented in
# Frazier,Powell,Dayanik 2009 on page 14].  This gives,
#   s*normcdf(s) + normpdf(s) = normpdf(s)*[1-s^2/(s^2+1)] = normpdf(s)/(s^2+1).
	n = len(s)
	s = np.array(s)
	logy = np.array([None]*n)
	index = [i for i in range(n) if s[i] < -10]
	if len(index) > 0:
		logy[index] = np.subtract(LogNormPDF(s[index]), np.log(np.add(np.power(s[index], 2), 1).astype(float)))

	index = [i for i in range(n) if s[i] >= -10]
	if len(index) > 0:
		s_norm_cdf = [norm.cdf(s[i]) for i in index]
		s_norm_pdf = [norm.pdf(s[i]) for i in index]
		logy[index] = np.log(np.add(np.multiply(s[index], s_norm_cdf), s_norm_pdf).astype(float))
	return logy


# logy = LogNormPDF(z)
# Returns the log of the normal pdf evaluated at z.  z can be a vector or a scalar.
def LogNormPDF(z):

	cons = -0.5*np.log(2*np.pi)
	logy = cons - np.divide(np.power(z, 2), 2)
	return logy


# function y=LogSumExp(x)
# Computes log(sum(exp(x))) for a vector x, but in a numerically careful way.
def LogSumExp(x):
	xmax = np.max(x)
	diff_max = x-xmax
	y = xmax + np.log(np.sum(np.exp(diff_max.astype(float))))
	return y



    
