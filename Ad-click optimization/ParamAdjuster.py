import pandas as pd
import time
from simulator import Simulator
from Policies.AdClickPolicy import Policy_AdClickPolicy as acp
from auction import Auction
import glob
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

def get_params():
    sheet = pd.read_excel("Parameter_Values.xlsx", sheet_name="Sheet1") # ensure that the sheet name is correctly inputted
    params = zip(sheet['theta_b'], sheet['w'])
    return list(params)

def param_iter(p):
    # change parameter values in AdClickPolicy
    acp.theta_b = p[0]
    acp.w = p[1]
    print("theta_b={}, w={}".format(acp.theta_b, acp.w))

    # run simulator with altered AdClickPolicy
    t_start = time.time()
    print("{:.2f} sec: start loading simulator".format(time.time() - t_start))
    sim = Simulator()

    param, attrs = Auction.read_init_xlsx("auction_ini_02.xlsx")
    auc = Auction(param, attrs)
    aucts = auc.generate_sample()
    # aucts = sl.load_auction_p("auction_01.p")   # loading from a snapshot example.
    sim.read_in_auction(aucts)

    print("{:.2f} sec: finished loading simulator".format(time.time() - t_start))
    for t in range(param['max iteration']):
        sim_res = sim.step()
        print("{:.2f} sec: simulation iter {}, auction happened? {}".format(time.time() - t_start, t, sim_res))

    sim.output_hist_to_xlsx("output_master_aggregate_{}.xlsx".format(p))
    print("{:.2f} sec: created output files.".format(time.time() - t_start))
    return True

def aggregate_data():
    all_data = pd.DataFrame()

    for f in glob.glob("../adclick_simulator/output_master_aggregate*.xlsx"):
        #file_name = pd.DataFrame("{}".format(f))
        df = pd.read_excel(f)
        #all_data = all_data.append(file_name, ignore_index=True)
        all_data = all_data.append(df, ignore_index=True)

    # we want to collect only the final results, which are in the last row of every Excel file
    select_data = all_data.iloc[2351::2352, :]
    return select_data

if __name__ == "__main__":
    params = get_params()
    print(params)

    # we use multiprocessing to run the simulator across 3 different set of parameters at the same time
    pool = mp.Pool(processes=3)
    pool.map(param_iter, params)
    results = aggregate_data()
    # we can change the name of this file each time we run an iteration
    results.to_excel('tuning_7_24_PS_only_2_3.xlsx', sheet_name='Sheet1')