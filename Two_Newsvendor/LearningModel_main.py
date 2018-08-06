"""
Two Newsvendor as a Learning Problem 

Author: Andrei Graur

This program simulates the game with the model implemented in 
TwoNewsvendorLearning.py

Run the code with the python command, no arguments given. 

"""

import numpy as np
import pandas as pd
import math
import xlrd

from TwoNewsvendor import Model_Field
from TwoNewsvendor import Model_Central
from TwoNewsvendorLearning import Choice
from TwoNewsvendorLearning import Learning_model_field
from TwoNewsvendorLearning import Learning_model_central

if __name__ == "__main__":

	# read from the spreadsheet
	raw_data = pd.read_excel("NewsvendorGame.xlsx", sheet_name="Table")
	nb_rounds = len(raw_data.index) - 1

	cost_data = pd.read_excel("NewsvendorGame.xlsx", sheet_name="Parameters")
	o_q = cost_data['Values'][0]
	u_q = cost_data['Values'][1]
	o_qPrime = cost_data['Values'][2]
	u_qPrime = cost_data['Values'][3]

	state_names_field = ['estimate', 'source_bias', 'source_var',
						'central_bias', 'central_var']
	decision_names_field = ['quantity_requested']

	state_names_central = ['field_request', 'field_bias', 'field_var']
	decision_names_central = ['quantity_allocated']

	t = 0
	total_pen_field = 0
	total_pen_central = 0
	# initialize the lists meant for the output colums in the spreadsheet
	demand_List = []
	allocated_q_List = []
	cost_field_List = []
	cost_central_List = []
	biases_field = []
	biases_central = []

	# do the simulation for the first round

	estimate = raw_data['Initial_Estimates'][t]
	# initialize the state of the field agent 
	state_field = {'estimate': estimate, 'source_bias': 0,
				  'source_var': (estimate * 0.2) ** 2, 'central_bias': 0,
				  'central_var': (estimate * 0.2) ** 2, "o_q": o_q, "u_q": u_q}

	DEFAULT_UCB_PARAM = 4
	M_field = Learning_model_field(DEFAULT_UCB_PARAM, state_names_field, decision_names_field, state_field)

	# choose what bias to add
	maxUCB = - np.inf
	bias_field = -1
	for x in range(11):
		if M_field.choices[x].get_UCB_value(1) >= maxUCB:
			maxUCB = M_field.choices[x].get_UCB_value(1)
			bias_field = x
	decision_field = (round(state_field['estimate'] - 
					 state_field['source_bias'] + bias_field))
	M_field.build_decision({'quantity_requested': decision_field})

	# initialize the state of the central command 
	state_central = {'field_request': decision_field, 'field_bias': 0,
					'field_var': (decision_field * 0.2) ** 2, "o_qPrime": o_qPrime,
					"u_qPrime": u_qPrime}
	M_central = Learning_model_central(DEFAULT_UCB_PARAM, state_names_central, decision_names_central, state_central)
	
	# choose what bias to add
	maxUCB = - np.inf
	bias_central = 1
	for x in range(-11, 1):
		if M_central.choices[x].get_UCB_value(1) >= maxUCB:
			maxUCB = M_central.choices[x].get_UCB_value(1)
			bias_central = x
	decision_central = (round(state_central['field_request'] - 
					   state_central['field_bias'] + bias_central))
	M_central.build_decision({'quantity_allocated': decision_central})
	demand = raw_data['Demand'][t]

	# see what penalties were incurred and update things 
	exog_info_field = {'allocated_quantity': decision_central, 'demand': demand}
	pen_field = M_field.objective_fn(decision_field, exog_info_field)
	pen_central = M_central.objective_fn(decision_central, demand)
	# save the relevant information for the output 
	demand_List.append(demand)
	allocated_q_List.append(decision_central)
	cost_field_List.append(pen_field)
	cost_central_List.append(pen_central)
	biases_field.append(bias_field)
	biases_central.append(bias_central)
	
	total_pen_field += pen_field
	total_pen_central += pen_central
	prev_field_bias = decision_field - demand 

	for t in range(1, nb_rounds):
		estimate = raw_data['Initial_Estimates'][t]
		# update the state of the field agent
		state_field = M_field.transition_fn(state_field, pen_field, bias_field, demand, estimate)
		# choose what bias to add
		maxUCB = - np.inf
		bias_field = -1
		for x in range(11):
			if M_field.choices[x].get_UCB_value(t + 1) >= maxUCB:
				maxUCB = M_field.choices[x].get_UCB_value(t + 1)
				bias_field = x
		decision_field = (round(state_field['estimate'] - 
						 state_field['source_bias'] + bias_field))
		
		M_field.build_decision({'quantity_requested': decision_field})
		# update the state of the central command 
		state_central = M_central.transition_fn(state_central, pen_central, bias_central, decision_field, prev_field_bias)
		
		# choose what bias to add
		maxUCB = - np.inf
		bias_central = -1
		for x in range(-11, 1):
			if M_central.choices[x].get_UCB_value(t + 1) >= maxUCB:
				maxUCB = M_central.choices[x].get_UCB_value(t + 1)
				bias_central = x
		decision_central = (round(state_central['field_request'] + bias_central))
		M_central.build_decision({'quantity_allocated': decision_central})
		
		central_bias = decision_central - decision_field
		demand = raw_data['Demand'][t]
		# update the penalties 
		exog_info_field = {'allocated_quantity': decision_central, 'demand': demand}
		
		pen_field = M_field.objective_fn(decision_field, exog_info_field)
		pen_central = M_central.objective_fn(decision_central, demand)
		# save the relevant information for the output 
		demand_List.append(demand)
		allocated_q_List.append(decision_central)
		cost_field_List.append(pen_field)
		cost_central_List.append(pen_central)
		biases_field.append(bias_field)
		biases_central.append(bias_central)

		total_pen_field += pen_field
		total_pen_central += pen_central
		prev_field_bias = decision_field - demand 

	output = pd.DataFrame({'demand': demand_List, 'allocated_quantity': allocated_q_List, 
			'cost field': cost_field_List, 'cost central': cost_central_List, 
			'biases_field': biases_field, 'biases_central': biases_central})
	output.to_excel("LearningNewsvendorGameOutput.xlsx", sheet_name="Output", index = False)
	print("Total penalty for field was %f." % total_pen_field)
	print("Total penalty for central was %f." % total_pen_central)

	pass
