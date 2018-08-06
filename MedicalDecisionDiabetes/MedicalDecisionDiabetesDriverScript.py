"""
Driver Script - Medical Decisions Diabetes Treatment

"""     

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from MedicalDecisionDiabetesModel import MedicalDecisionDiabetesModel as MDDM
from MedicalDecisionDiabetesModel import Beta
from MedicalDecisionDiabetesPolicy import MDDMPolicy
        
# unit testing
if __name__ == "__main__":
    '''
    this is an example of creating a model, choosing the decision according to the policy of choice, 
    and running the model for a fixed time period (in months)
    '''
    
    file = 'MDDMparameters.xlsx'
    S0 = pd.read_excel(file, sheet_name = 'parameters1')
    additional_params = pd.read_excel(file, sheet_name = 'parameters2')
    
    state_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
    
    # for each drug: (first entry: mu_0, second entry: beta_0, third entry: number of times drug x was given)
    init_state = {'M': [ S0.loc['M', 'mu'] , Beta(S0.loc['M', 'sigma']), 0], \
                  'Sens': [S0.loc['Sens', 'mu'] , Beta(S0.loc['Sens', 'sigma']), 0], \
                  'Secr': [S0.loc['Secr', 'mu'] , Beta(S0.loc['Secr', 'sigma']), 0], \
                  'AGI': [S0.loc['AGI', 'mu'] , Beta(S0.loc['AGI', 'sigma']), 0], \
                  'PA': [S0.loc['PA', 'mu'] , Beta(S0.loc['PA', 'sigma']), 0]}
    
    # in order: Metformin, Sensitizer, Secretagoge, Alpha-glucosidase inhibitor, Peptide analog.
    x_names = ['M', 'Sens', 'Secr', 'AGI', 'PA']
    policy_names = ['UCB', 'IE', 'PureExploitation', 'PureExploration']

    Model = MDDM(state_names, x_names, init_state)
    # initialize sigma_w
    Model.sigma_w = additional_params.loc['sigma_w', 0]
    P = MDDMPolicy(Model, policy_names)
    
    # Initialize
    # each time step is 1 month.
    t_stop = int(additional_params.loc['N', 0]) # number of times we test the drugs
    L = int(additional_params.loc['L', 0]) # number of samples
    theta_range_1 = np.arange(additional_params.loc['theta_start', 0],\
                              additional_params.loc['theta_end', 0],\
                              additional_params.loc['increment', 0])
    
    # make a dictionary with the objectives as lists
    theta_obj = {'UCB': [], 
                 'IE': []}
    
    # Here we are going to run each policy separately
    
# =============================================================================
#    policy: IE
# =============================================================================
    
# loop over theta (theta)
    IE_start = time.time()
    for theta in theta_range_1:
        # want to generate a dictionary of thetas to the run av?
        # print("theta = {}".format(theta))
        # loop over samples of mu (the truth), and epsilon (the noise) (L)
        obj_dict_llevel = []
        for l in range(1,L+1):
            # loop over time (N, in notes)
            obj_entry_nlevel = []
            # print("l = {}".format(l))
            for n in range(t_stop):
                current_state_dict = {}
                for i in state_names:
                    current_state_dict[i] = getattr(Model.state, i)
                # policy: IE
                # make decision based on chosen policy
                decision = P.IE(current_state_dict, Model.t, theta)
                # update model according to the decision
                # print("t = {}, \nstate = {} \ndecision = {}, \nobjective = {} \n".format(Model.t, current_state_dict, decision, Model.obj)) # better format state
                obj_entry_nlevel.append(Model.obj) #[]
                Model.step(decision)
                Model.t_update()
            obj_dict_llevel.append(obj_entry_nlevel) # creates l sample runs of the n timesteps, stores this in the list in dictionary
            # print(obj_dict_llevel)
            Model.t = 0 # reset the time counter
            Model.obj = 0 #reset the objective for a new trial
        obj_dict_llevel_sum = [sum(elts) for elts in zip(*obj_dict_llevel)] # sums the l sample runs into a new list
        run_av_obj_dict_llevel_sum= [entry / L for entry in obj_dict_llevel_sum] # average objective of L samples (as a list, each entry corresponds to time)
        theta_obj['IE'].append(run_av_obj_dict_llevel_sum[-1])
    IE_end = time.time()
    print("{} secs".format(IE_end - IE_start))

# =============================================================================
#     Policy: UCB
# =============================================================================

# for UCB we need to initialize the no. of times we have administered the drug because this value is being used in the policy.
    UCB_start = time.time()
    for theta in theta_range_1:
        # want to generate a dictionary of thetas to the run av?
        # print("theta = {}".format(theta))
        # loop over samples of mu (the truth), and epsilon (the noise) (L)
        obj_dict_llevel = []
        for l in range(1,L+1):
            # loop over time (N, in notes)
            obj_entry_nlevel = []
            # print("l = {}".format(l))
            for n in range(t_stop):
                current_state_dict = {}
                for i in state_names:
                    current_state_dict[i] = getattr(Model.state, i)
                # policy: UCB
                # make decision based on chosen policy
                decision = P.UCB(current_state_dict, Model.t, theta)
                # update model according to the decision
                # print("t = {}, \nstate = {} \ndecision = {}, \nobjective = {} \n".format(Model.t, current_state_dict, decision, Model.obj)) # better format state
                obj_entry_nlevel.append(Model.obj) #[]
                Model.step(decision)
                Model.t_update()
            obj_dict_llevel.append(obj_entry_nlevel) # creates l sample runs of the n timesteps, stores this in the list in dictionary
            # print(obj_dict_llevel)
            Model.t = 0 # reset the time counter
            Model.obj = 0 #reset the objective for a new trial
        obj_dict_llevel_sum = [sum(elts) for elts in zip(*obj_dict_llevel)] # sums the l sample runs into a new list
        run_av_obj_dict_llevel_sum= [entry / L for entry in obj_dict_llevel_sum] # average objective of L samples (as a list, each entry corresponds to time)
        theta_obj['UCB'].append(run_av_obj_dict_llevel_sum[-1])
    UCB_end = time.time()
    print("{} secs".format(UCB_end - UCB_start))


# =============================================================================
#     Generating Plots
# =============================================================================
    
    fig = plt.figure()
    plt.title('Comparison of IE and UCB for the Medical Decisions Diabetes Model: \n (N = {}, L = {}, Theta range: ({}, {}, {}))'.format(t_stop, L, theta_range_1[0], theta_range_1[-1], theta_range_1[1] - theta_range_1[0]))
    plt.plot(theta_range_1, theta_obj['IE'], label = "IE")
    plt.scatter(theta_range_1, theta_obj['IE'])
    plt.plot(theta_range_1, theta_obj['UCB'], label = "UCB")
    plt.scatter(theta_range_1, theta_obj['UCB'])
    #plt.plot(t, obj_dict['PureExploitation'], label = "Pure Exploitation objective")
    #plt.plot(t, obj_dict['PureExploration'], label = "Pure Exploration objective")
    plt.legend()
    plt.xlabel('theta')
    plt.ylabel('estimated value (F^l)')
    plt.show()
    fig.savefig('UCB_IE_Comparison_test.jpg')