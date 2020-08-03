import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import pdb
import main_simulation_bursty as main_sim_bursty

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import sim_analyze as sim_anal  


data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/" 


#seeds = np.random.randint(0,9999999,1) # Either generate the seeds everytime, or generate them once, save them and resuse for them for reproducibility
seeds = [234]

#pickle.dump(seeds,open(path5+"seeds.pickle","w"))
# Set the seed path
seed_path = ""

subject = sys.argv[1]
state = sys.argv[2]
channel = sys.argv[3]
gpe_rat = sys.argv[4]
gpe_ratio = sys.argv[5]
stn_ratio = sys.argv[6]

piece = pickle.load(open(data_target_dir+"piece_wise_rate.pickle","rb"))


#def run_subject_simulation(subject,state,channel,seeds):
for seed in seeds:

    ts = piece[subject][state][channel]         
    
    sim_name = "rateEffect"
    params = dict()
    params["stn_inp"] = ts
    params["seed"] = seed
    params["simtime"] = 10000
    params["gpe_inp"] = gpe_rat
    params["gpe_ratio"] = gpe_ratio
    params["stn_ratio"] = stn_ratio

    params["name"] = subject+"_"+state+"_gpe_ratio_"+str(gpe_ratio)+"_stn_ratio_"+str(stn_ratio) #+"_"+channel
    sim_name = sim_name+"_"+str(subject)+"_"+str(seed)
    main_sim_bursty.runSim(params)





