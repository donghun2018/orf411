"""
Author: Andrei Graur

This is the code that uses the first Two Newsvendor model. 
Run the code with the python command, no arguments given. 

"""

from collections import namedtuple
from TwoNewsvendor import Model_Field
from TwoNewsvendor import Model_Central

import numpy as np
import pandas as pd
import math
import xlrd

if __name__ == "__main__":

    # read from the spreadsheet
    raw_data = raw_data = pd.read_excel("NewsvendorGame.xlsx", sheet_name="Table")
    nb_rounds = len(raw_data.index) - 1

    parameters = pd.read_excel("NewsvendorGame.xlsx", sheet_name="Parameters")
    alpha_qepsilon = parameters['Values'][4]
    alpha_qqPrime = parameters['Values'][5]
    alpha_qPrimeq = parameters['Values'][6]

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

    # do the simulation for the first round
    estimate = raw_data['Initial_Estimates'][t]
    # initialize the state of the field agent 
    state_field = {'estimate': estimate, 'source_bias': 0,
                       'source_var': (estimate * 0.2) ** 2, 'central_bias': 0,
                       'central_var': (estimate * 0.2) ** 2, "o_q": 2, "u_q": 10}
    
    M_field = Model_Field(state_names_field, decision_names_field, state_field)
    decision_field = (round(state_field['estimate'] - state_field['source_bias'] + 
                     alpha_qepsilon * math.sqrt(state_field['source_var']) - 
                     state_field['central_bias'] + 
                     alpha_qqPrime * math.sqrt(state_field['central_var'])))
    
    M_field.build_decision({'quantity_requested': decision_field})
    # initialize the state of the central command 
    state_central = {'field_request': decision_field, 'field_bias': 0,
                       'field_var': (decision_field * 0.2) ** 2, "o_qPrime": 5,
                    "u_qPrime": 1}
    
    M_central = Model_Central(state_names_central, decision_names_central, state_central)
    decision_central = (round(state_central['field_request'] - 
                       state_central['field_bias'] + 
                       alpha_qPrimeq * math.sqrt(state_central['field_var'])))
    
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

    total_pen_field += pen_field
    total_pen_central += pen_central
    prev_field_bias = decision_field - demand 
    

    for t in range(1, nb_rounds):
        estimate = raw_data['Initial_Estimates'][t]
        # update the state of the field agent
        state_field = M_field.transition_fn(state_field, central_bias, demand, estimate)
        decision_field = (round(state_field['estimate'] - state_field['source_bias'] 
                         + alpha_qepsilon * math.sqrt(state_field['source_var']) - 
                         state_field['central_bias'] + 
                         alpha_qqPrime * math.sqrt(state_field['central_var'])))
        M_field.build_decision({'quantity_requested': decision_field})
        # update the state of the central command 
        state_central = M_central.transition_fn(state_central, decision_field, prev_field_bias)
        decision_central = (round(state_central['field_request'] - 
                           state_central['field_bias'] + 
                           alpha_qPrimeq * math.sqrt(state_central['field_var'])))
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

        total_pen_field += pen_field
        total_pen_central += pen_central
        prev_field_bias = decision_field - demand 

    output = pd.DataFrame({'demand': demand_List, 'allocated_quantity': allocated_q_List, 'cost field': cost_field_List, 'cost central': cost_central_List})
    output.to_excel("NewsvendorGameOutput.xlsx", sheet_name="Output", index = False)
    print("Total penalty for field was %f." % total_pen_field)
    print("Total penalty for central was %f." % total_pen_central)


    pass
