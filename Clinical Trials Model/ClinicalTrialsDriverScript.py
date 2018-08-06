"""
Clinical Trials Driver Script class

Raluca Cobzaru (c) 2018

"""

from collections import namedtuple
import numpy as np
import scipy
import pandas as pd
from ClinicalTrialsModel import ClinicalTrialsModel
from ClinicalTrialsPolicy import ClinicalTrialsPolicy

if __name__ == "__main__":
	# initializes a policy object and a model object, then runs the policy on the model
	policy_names = ['model_A', 'model_B', 'model_C', 'model_C_extension']
	state_names = ['potential_pop', 'success', 'failure', 'l_response']
	# extracts data from given data set; defines initial state
	file = 'Trials Parameters.xlsx'
	raw_data = pd.ExcelFile(file)
	data = raw_data.parse('Exogenous Data')
	initial_state = {'potential_pop': float(data.iat[0, 0]),
					 'success': data.iat[1, 0],
					  'failure': float(data.iat[2, 0]),
					  'l_response': float(data.iat[3, 0]),
					  'theta_stop_low': data.iat[4, 0],
					  'theta_stop_high': data.iat[5, 0],
					  'alpha': data.iat[6, 0],
					  'K': int(data.iat[7, 0]),
					  'N': int(data.iat[8, 0]),
					  'trial_size': int(data.iat[9, 0]),
					  'patient_cost': data.iat[10, 0],
					  'program_cost': data.iat[11, 0],
					  'success_rev': data.iat[12, 0],
					  'sampling_size': int(data.iat[13, 0]),
					  'enroll_min': int(data.iat[14, 0]),
					  'enroll_max': int(data.iat[15, 0]),
					  'enroll_step': int(data.iat[16, 0]),
					  'H': int(data.iat[17, 0]),
					  'true_l_response': data.iat[18, 0],
					  'true_succ_rate': data.iat[19, 0]}
	decision_names = ['enroll', 'prog_continue', 'drug_success']
	
	# an example of running the Model B policy using the ClinicalTrialsPolicy run_policy function
	M = ClinicalTrialsModel(state_names, decision_names, initial_state, False)
	P = ClinicalTrialsPolicy(M, policy_names)
	t = 0
	stop = False
	policy_info = {'model_A': [-1, stop],
					'model_B': [-1, stop],
					'model_C': [-1, stop],
					'model_C_extension': [-1, stop]}
					
	P.run_policy(policy_info, "model_B", t)
	
	# alternate example of running the Model C policy without the run_policy function
	M = ClinicalTrialsModel(state_names, decision_names, initial_state, False)
	P = ClinicalTrialsPolicy(M, policy_names)
	t = 0
	stop = False
	policy_info = {'model_A': [-1, stop],
					'model_B': [-1, stop],
					'model_C': [-1, stop],
					'model_C_extension': [-1, stop]}
	policy = "model_C"
		
	while t <= M.initial_state['trial_size'] and policy_info[policy][1] == False: 
		# build decision policy
		p = P.build_policy(policy_info)
		
		# implements sampled distribution for p_true
		p_true_samples = np.random.beta(M.state.success, M.state.failure, M.initial_state['K'])
		p_belief = M.state.success / (M.state.success + M.state.failure)
	
		# drug_success = 1 if successful, 0 if failure, -1 if continue trial (for all policies)
		if p_belief > M.initial_state['theta_stop_high']:
			decision = {'prog_continue': 0, 'drug_success': 1}
			policy_info[policy][1] = True
		elif p_belief < M.initial_state['theta_stop_low']:
			decision = {'prog_continue': 0, 'drug_success': 0}
			policy_info[policy][1] = True
		else:
			decision = {'prog_continue': 1, 'drug_success': -1}
		
		# makes enrollment decision based on chosen policy
		if policy == "model_A":
			decision['enroll'] = P.model_A_policy(M.state, p.model_A)
		elif policy == "model_B":
			decision['enroll'] = P.model_B_policy(M.state, p.model_B)
		elif policy == "model_C_extension":
			decision['enroll'] = P.model_C_extension_policy(M.state, p.model_C_extension)
		elif policy == "model_C":
			decision['enroll'] = P.model_C_policy(M.state, p.model_C, time)
		
		x = M.build_decision(decision)
		print("t={}, obj={}, state.potential_pop={}, state.success={}, state.failure={}, x={}".format(t, M.objective, 
																									M.state.potential_pop, 
																									M.state.success, 
																									M.state.failure, x))
		# steps the model forward one iteration
		M.step(x)
		# updates policy info
		policy_info[policy][0] = decision['drug_success']
		# increments time
		t += 1
	print("t={}, obj={}, state.potential_pop={}, state.success={}, state.failure={}".format(t, M.objective, 
																								M.state.potential_pop, 
																								M.state.success, 
																								M.state.failure))
	
	pass