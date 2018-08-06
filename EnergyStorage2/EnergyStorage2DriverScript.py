from collections import namedtuple
import numpy as np
import math
from Forecast import Forecast
from EnergyStorage2Model import EnergyStorage2Model
from EnergyStorage2Policy import EnergyStorage2Policy
import pandas as pd
import copy

if __name__ == "__main__":

	forecast_names = ['f_L', 'f_W', 'f_P', 'f_G']
	state_names = [k for k in forecast_names]
	state_names.append('R')
	x_names = ['x_wr', 'x_wl', 'x_gr', 'x_rg', 'x_gl', 'x_rl', 'x_loss']
	
	# read in variables from excel file
	file = 'parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('parameters')
	beta = data.iat[0, 2]
	sigma_L = data.iat[1, 2]
	sigma_P = data.iat[2, 2]
	sigma_G = data.iat[3, 2]
	sigma_W = data.iat[4, 2]
	horizon = np.rint(data.iat[5, 2]).astype(int)
	price = data.iat[6, 2]
	a = data.iat[7, 2]
	b = data.iat[8, 2]
	eta = data.iat[9, 2]
	R_max = data.iat[10, 2]
	u_charge = data.iat[11, 2]
	u_discharge = data.iat[12, 2]
	iteration = 1
	trial_size = np.rint(data.iat[13, 2]).astype(int)
	zeta = data.iat[14, 2]
	step_size = data.iat[15, 2]
	step_rule = 1.0
	tolerance = data.iat[16, 2]
	seed = np.rint(data.iat[17, 2]).astype(int)
	prng = np.random.RandomState(seed)
	compare_trial_size = np.rint(data.iat[18, 2]).astype(int)
	
	# create forecasts
	forecast_L = Forecast(horizon, beta, sigma_L, type = 'L')
	forecast_P = Forecast(horizon, beta, sigma_P, type = 'P', price = price)
	forecast_G = Forecast(horizon, beta, sigma_G, type = 'G', forecast_L = forecast_L, a = a, b = b)
	forecast_W = Forecast(horizon, beta, sigma_W, type = 'W')
	forecasts = {'f_L' : forecast_L, 'f_W' : forecast_W, 'f_P' : forecast_P, 'f_G' : forecast_G}
	
	# define initial state
	init_state = {'R' : 0, 'f_L' : forecast_L.return_forecast(), 'f_W' : forecast_W.return_forecast(), 'f_P' : forecast_P.return_forecast(), 'f_G' : forecast_G.return_forecast()}
	
	# define initial theta
	theta_R = [1.0 for i in range(horizon)]
	theta_L = [1.0 for i in range(horizon)]
	theta_W = [1.0 for i in range(horizon)]
	theta = {'R': theta_R, 'L': theta_L, 'W': theta_W}
	
	# create model and copy to compute numerical derivative
	M = EnergyStorage2Model(state_names, x_names, forecast_names, forecasts, eta, init_state)
	M_prime = copy.deepcopy(M)
	
	# create copies to test overall performance at the end
	M1 = copy.deepcopy(M)
	M2 = copy.deepcopy(M)
	
	last_sign = 0
	theta1 = copy.deepcopy(theta)
	
	while iteration <= trial_size:
		# match forecasts
		M_prime.forecasts['f_L'].x_forecast = M.forecasts['f_L'].x_forecast
		M_prime.forecasts['f_W'].x_forecast = M.forecasts['f_W'].x_forecast
		M_prime.forecasts['f_P'].x_forecast = M.forecasts['f_P'].x_forecast
		M_prime.forecasts['f_G'].x_forecast = M.forecasts['f_G'].x_forecast
		
		M_prime.forecasts['f_L'].current_t = M.forecasts['f_L'].current_t
		M_prime.forecasts['f_W'].current_t = M.forecasts['f_W'].current_t
		M_prime.forecasts['f_P'].current_t = M.forecasts['f_P'].current_t
		M_prime.forecasts['f_G'].current_t = M.forecasts['f_G'].current_t
		
		# add random vector to approximate gradient
		unit_vector_R = np.concatenate([[0], prng.normal(size = horizon - 1)])
		unit_vector_L = np.concatenate([[0], prng.normal(size = horizon - 1)])
		unit_vector_W = np.concatenate([[0], prng.normal(size = horizon - 1)])
		theta_prime = copy.deepcopy(theta)
		theta_prime['R'] += unit_vector_R * zeta
		theta_prime['L'] += unit_vector_L * zeta
		theta_prime['W'] += unit_vector_W * zeta
		
		# run a trial
		for k in range(horizon):
			decision_M = EnergyStorage2Policy(M, theta, u_charge, u_discharge, eta, R_max, x_names).construct_decision()
			decision_M_prime = EnergyStorage2Policy(M_prime, theta_prime, u_charge, u_discharge, eta, R_max, x_names).construct_decision()
			M.step(decision_M)
			M_prime.step(decision_M_prime)
			
		obj_difference = M_prime.obj - M.obj
		print('{}, {}, {}'.format(iteration, M.obj, obj_difference))
		
		#update values
		theta['R'] += step_size / step_rule * obj_difference / zeta * unit_vector_R
		theta['L'] += step_size / step_rule * obj_difference / zeta * unit_vector_L
		theta['W'] += step_size / step_rule * obj_difference / zeta * unit_vector_W
		
		# reset relevant values
		M.obj = 0
		M_prime.obj = 0
		M.resetR()
		M_prime.resetR()
		iteration += 1
		
		# Kesten's rule
		if last_sign * obj_difference < 0:
			step_rule += 1
		last_sign = obj_difference
	
	print(theta)
	
	# compare performance
	for i in range(compare_trial_size):
		for k in range(horizon):
			decision1 = EnergyStorage2Policy(M1, theta1, u_charge, u_discharge, eta, R_max, x_names).construct_decision()
			decision2 = EnergyStorage2Policy(M2, theta, u_charge, u_discharge, eta, R_max, x_names).construct_decision()
			M1.step(decision1)
			M2.step(decision2)
		M1.resetR()
		M2.resetR()
	print('original : {}, updated: {}'.format(M1.obj / trial_size, M2.obj / trial_size))