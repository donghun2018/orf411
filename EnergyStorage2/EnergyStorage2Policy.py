"""
Energy Storage II Policy

"""
from collections import namedtuple
import numpy as np
from cvxopt import matrix, printing, solvers
from scipy.optimize import linprog
from EnergyStorage2Model import EnergyStorage2Model

class EnergyStorage2Policy():
	"""
	Base class for Energy Storage II policy
	"""

	def __init__(self, Model, theta, u_charge, u_discharge, eta, R_max, x_names):
		"""
		Initializes the policy

		:param Model: EnergyStorage2Model - model to construct decision for
		:param theta: dict - thetas needed for parametrization of lookahead policy
		:param u_charge: float - maximum energy that can flow into the battery
		:param u_charge: float - maximum energy that can flow out of the battery
		:param eta: float - battery efficiency
		:param R_max: float - battery capacity
		:param x_names: list(str) - decision variable dimension names
		"""

		self.x_names = x_names
		self.M = Model
		self.theta = theta
		self.u_charge = u_charge
		self.u_discharge = u_discharge
		self.eta = eta
		self.R_max = R_max

	# returns decision based on a parametrized lookahead policy
	def construct_decision(self):
		# Solve linear program
		# max  cx
		# s.t. Ax = b
		#	  Fx <= g
		#
		c = []
		A = []
		F = []
		b = []
		g = []

		# x is a list of R_tt', x_wr_tt', x_gr_tt', x_rg_tt', x_gl_tt', x_rl_tt', x_loss_tt' for t' from t to t + horizon - 1 and R_t(t + horizon)

		# construct required matrices
		if len(self.M.state.f_P) > 0:
			for i in range(len(self.M.state.f_P)):
				c.extend([0, 0, self.M.state.f_P[i], -self.M.state.f_G[i], self.eta * self.M.state.f_G[i], self.M.state.f_P[i] - self.M.state.f_G[i], self.eta * self.M.state.f_P[i], 0])
				F_new = [[] for i in range(14)]
				A_new = [[]]
				# no variables in this range
				for j in range(i):
					for k in range(14):
						F_new[k].extend([0, 0, 0, 0, 0, 0, 0, 0])
					A_new[0].extend([0, 0, 0, 0 ,0 ,0 ,0 ,0])
				# relevant variables here
				F_new[0].extend([0, 0, 1, 0, 0, 1, self.eta, 1])
				F_new[1].extend([-self.theta['R'][i], 0, 0, 0, 1, 0, 1, 0])
				F_new[2].extend([1/self.eta, 1, 0, 1, 0, 0, 0, 0])
				F_new[3].extend([0, 1, 1, 0, 0, 0, 0, 0])
				F_new[4].extend([0, 1, 0, 1, 0, 0, 0, 0])
				F_new[5].extend([0, 0, 0, 0, 1, 0, 1, 0])
				F_new[6].extend([-1, 0, 0, 0, 0, 0, 0, 0])
				F_new[7].extend([0, -1, 0, 0 ,0 ,0 ,0 ,0])
				F_new[8].extend([0, 0, -1, 0 ,0 ,0 ,0 ,0])
				F_new[9].extend([0, 0, 0, -1 ,0 ,0 ,0 ,0])
				F_new[10].extend([0, 0, 0, 0 ,-1 ,0 ,0 ,0])
				F_new[11].extend([0, 0, 0, 0 ,0 ,-1 ,0 ,0])
				F_new[12].extend([0, 0, 0, 0 ,0 ,0 ,-1 ,0])
				F_new[13].extend([0, 0, 0, 0 ,0 ,0 ,0 ,-1])
				A_new[0].extend([1, self.eta, 0, self.eta, -1, 0, -1, 0, -1])
				# add remainder of variables
				for j in range(i + 1, len(self.M.state.f_P)):
					for k in range(14):
						F_new[k].extend([0, 0, 0, 0, 0, 0, 0, 0])
					A_new[0].extend([0, 0, 0, 0 ,0 ,0 ,0 ,0])
				for k in range(14):
					F_new[k].extend([0])
				A.extend(A_new)
				F.extend(F_new)
				b.append(0.0)
				g.extend([self.theta['L'][i] * self.M.state.f_L[i], 0, self.R_max / self.eta, self.theta['W'][i] * self.M.state.f_W[i], self.u_charge, self.u_discharge])
				g.extend([0 for k in range(14 - 6)])
			c.extend([0])
			# we want to maximize not minimize
			c = -1 * np.array(c)
			# require x[0] = R_00
			A_new = [[]]
			A_new[0].extend([0] * (8 * len(self.M.state.f_P) + 1))
			A_new[0][0] = 1
			A.extend(A_new)
			b.append(self.M.state.R)
			
			# transform into matrices for cvxopt
			c_prime = matrix(np.array([c]).T)
			F_prime = matrix(np.array(F))
			A_prime = matrix(np.array(A))
			g_prime = matrix(np.array([g]).T)
			b_prime = matrix(np.array([b]).T)
			
			# solve with glpk
			solution = solvers.lp(c_prime, F_prime, g_prime, A_prime, b_prime, solver = 'glpk', options={'glpk':{'msg_lev': 'GLP_MSG_OFF', 'meth': 'GLP_PRIMAL', 'tm_lim': 10000, 'tol_bnd': 1e-4, 'tol_piv': 1e-6}})
			x = solution['x']
			# if glpk fails use linprog from scipy (much slower and imprecise)
			if solution['status'] != 'optimal':
				decision = linprog(c, A_ub = F, b_ub = g, A_eq = A, b_eq = b, options = {"disp": False, 'sparse': False, 'lstsq': False, 'tol': 10 ** (-12)}, method = 'interior-point')
				x = decision.x

			# build decision
			decision = {}
			for i in range(len(self.x_names)):
				decision[self.x_names[i]] = x[i + 1]
			decision = self.M.build_decision(decision)
		else:
			decision = {}
			for i in range(len(self.x_names)):
				decision[self.x_names[i]] = 0.0
			decision = self.M.build_decision(decision)
		return decision
