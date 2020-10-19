import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import pdb
sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/GA_scripts/")
import GA_params as GA_par
#import scipy.stats as stats

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/simulations/")

import main_simulation_ga_check as main_sim_ga_check

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import params_d_ssbn as params_d
import sim_analyze as sim_anal  

sys.path.append("/home/bahuguna/Work/Data_Alex/scripts/")
import analyze as anal
piece = pickle.load(open("/home/bahuguna/Work/Data_Alex/target_data/piece_wise_rate.pickle","rb"))

data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/GA_params/" 
seeds = [234]


subject = sys.argv[1]
seed = sys.argv[2]


path =  "/home/bahuguna/Work/Data_Alex/target_data/"+str(seed)+"/"+subject+"/"

comb = pickle.load(open(path+"Combinations_"+str(seed)+".pickle","rb"))
#comb = pickle.load(open(path+"Combinations_"+str(seed)+"_new_bckup.pickle","rb"))
spec_entropy = pickle.load(open("/home/bahuguna/Work/Data_Alex/target_data/spectral_entropies.pickle","rb"))
ctx_elec1,stn_elec1 = spec_entropy[subject]["CTX_fit"], spec_entropy[subject]["STN_fit"]


channel = ctx_elec1
for i in np.arange(len(comb["params"])):

    for st in ["OFF","ON"]:
        ts = piece[subject][st][channel]
        
        params = dict()
        params["stn_inp"] = ts 
        params["stn_bck_rate"] = comb["params"][i]["stn_bck_rate"] 
        #params["stn_bck_rate"] = 550 
        params["gpe_inp"] = comb["params"][i]["gpe_inp"]  
        #params["gpe_inp"] = 1400 
        #params["gpe_inp"] = 1300 
        params["simtime"] = 10000
        params["seed"] = int(seed)
        params["scale_gpe_inh"] = comb["params"][i]["scale_gpe_inh"] 
        #params["scale_gpe_inh"] = 0.1 
        #params["scale_gpe_inh"] = 0.7
        #params["scale_gpe_inh"] = 0.8
        #params["scale_stn_exc"] = 0.3
        params["scale_stn_exc"] = comb["params"][i]["scale_stn_exc"]
        params["scale_synaptic"] = comb["params"][i]["scale_synaptic"] 
        #else:
            #params["scale_synaptic"] = comb["params"][0]["scale_synaptic"][0] 
            #params["scale_synaptic"] = 0.3
            #params["scale_synaptic"] = 0.9
        #    params["scale_synaptic"] = 1.2
        params["scale_conn"] = comb["params"][i]["scale_conn"] 
        #params["scale_conn"] = 0.2 
        #params["scale_conn"] = 0.7
        params["scale_delays"] = comb["params"][i]["scale_delays"] 
        #params["scale_delays"] = 5.0 
        #params["scale_delays"] = 3.0
        #params["scale_delays"] = 2.0
        #params["sim_type"] = "bursty"
        params["sim_type"] = comb["params"][i]["sim_type"]
        #params["gpe_ratio"] = 0.2 
        params["gpe_ratio"] = comb["params"][i]["gpe_ratio"] 
        #params["stn_ratio"] = 0.4
        params["stn_ratio"] = comb["params"][i]["stn_ratio"]
        params["ip_stn_delay"] = comb["params"][i]["ip_stn_delay"]
        params["ip_gpe_delay"] = comb["params"][i]["ip_gpe_delay"]

        params["name"] = subject+"_"+st 
        params["path"] = data_target_dir 

        main_sim_ga_check.runSim(params)




