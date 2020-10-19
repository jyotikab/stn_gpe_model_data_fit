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

fig_target_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/GA_params/"
path =  "/home/bahuguna/Work/Data_Alex/target_data/"+str(seed)+"/"+subject+"/"

comb = pickle.load(open(path+"Combinations_"+str(seed)+".pickle","rb"))

#errors = [ comb["errors"][i]["ON"]["left"] +comb["errors"][i]["ON"]["right"]+comb["errors"][i]["OFF"]["left"]+comb["errors"][i]["OFF"]["right"]     for i in np.arange(len(comb["params"])) ]
#list_sort = np.argsort(errors).tolist()
#ind_plot = list_sort[:5] # 5 best solutions
subject_data = pickle.load(open("/home/bahuguna/Work/Data_Alex/target_data/Subject_data.pickle","rb"))
spec_entropy = pickle.load(open("/home/bahuguna/Work/Data_Alex/target_data/spectral_entropies.pickle","rb"))
ctx_elec1,stn_elec1 = spec_entropy[subject]["CTX_fit"], spec_entropy[subject]["STN_fit"]
channel = ctx_elec1

#comb = pickle.load(open(path+"Combinations_"+str(seed)+"_bckup.pickle","rb"))
#channel = "C3"
for i in np.arange(len(comb["params"])):
    #if i not in ind_plot:
    #    continue

    gpe_rat = float(comb["params"][i]["gpe_inp"])
    stn_rat = "["+str(float(comb["params"][i]["stn_bck_rate"]))+"]"
    sim_type = comb["params"][i]["sim_type"]
    gpe_ratio = comb["params"][i]["gpe_ratio"]
    stn_ratio = comb["params"][i]["stn_ratio"]
    scale_gpe_inh = comb["params"][i]["scale_gpe_inh"]
    scale_stn_exc = comb["params"][i]["scale_stn_exc"]
    scale_synaptic = comb["params"][i]["scale_synaptic"]
    scale_conn = comb["params"][i]["scale_conn"]
    scale_delays = comb["params"][i]["scale_delays"]
    ip_stn_delay = comb["params"][i]["ip_stn_delay"]
    ip_gpe_delay = comb["params"][i]["ip_gpe_delay"]

    chosen_elec = comb["correlation"][i]["ON"]["stn_electrode"]
    chosen_elec_mask = [ x in subject_data[subject]["OFF"].keys()   for x in chosen_elec]
    chosen_elec_final = np.array(chosen_elec)[chosen_elec_mask][0]


    path_fig = fig_target_dir+subject+"/" 
    if os.path.isdir(path_fig+str(i+1)) == False:
       os.mkdir(path_fig+str(i+1))

    for st in ["OFF","ON"]:
        cmd1 = "python plot_raster.py "+ subject+" "+ st+" "+channel+" "+ str(gpe_rat)+" "+ str(stn_rat)+" "+ sim_type+" "+ str(gpe_ratio)+" "+str(stn_ratio)+" "+str(scale_gpe_inh)+" "+str(scale_stn_exc)+" "+str(scale_synaptic)+" "+str(scale_conn)+" "+str(scale_delays)+" "+str(ip_stn_delay)+" "+str(ip_gpe_delay)+" "+str(i+1)+" "+chosen_elec_final+ "  y"
        os.system(cmd1)

        cmd2 = "python plot_fft.py "+ subject+" "+ st+" "+channel+" "+ str(gpe_rat)+" "+ str(stn_rat)+" "+ sim_type+" "+ str(gpe_ratio)+" "+str(stn_ratio)+" "+str(scale_gpe_inh)+" "+str(scale_stn_exc)+" "+str(scale_synaptic)+" "+str(scale_conn)+" "+str(scale_delays)+" "+str(ip_stn_delay)+" "+str(ip_gpe_delay)+" "+str(i+1)+" "+chosen_elec_final+"  y"
        os.system(cmd2)

        cmd3 = "python plot_subject_spectrogram_comparison.py "+ subject+" "+ st+" "+channel+" "+ str(gpe_rat)+" "+ str(stn_rat)+" "+ sim_type+" "+ str(gpe_ratio)+" "+str(stn_ratio)+" "+str(scale_gpe_inh)+" "+str(scale_stn_exc)+" "+str(scale_synaptic)+" "+str(scale_conn)+" "+str(scale_delays)+" "+str(ip_stn_delay)+" "+str(ip_gpe_delay)+" "+str(i+1)+" "+chosen_elec_final+"  y"
        os.system(cmd3)

