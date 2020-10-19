import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import pdb
import pylab as pl

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import sim_analyze as sim_anal  
import params_d_ssbn as params_d
pars = params_d.Parameters()

sys.path.append("/home/bahuguna/Work/Data_Alex/scripts/")
import analyze as anal

fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/"



data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/" 

STN_electrodes = ["LS"+str(i)  for i in np.arange(1,5)] +["L"+str(i)  for i in np.arange(1,5)]+["R"+str(i)  for i in np.arange(1,5)]+["RS"+str(i)  for i in np.arange(1,5)]
binw = 1000
subject = sys.argv[1]
state = sys.argv[2]
channel = sys.argv[3]
gpe_rat = sys.argv[4]
stn_rat = sys.argv[5]
sim_type = sys.argv[6]
gpe_ratio = sys.argv[7]
stn_ratio = sys.argv[8]
scale_gpe_inh = sys.argv[9]
scale_stn_exc = sys.argv[10]
scale_synaptic = sys.argv[11]
scale_conn = sys.argv[12]
scale_delays = sys.argv[13]
ip_stn_delay = sys.argv[14]
ip_gpe_delay = sys.argv[15]

index = sys.argv[16]
electrode = sys.argv[17]
ga = sys.argv[18]

seeds = [234]
if ga == "n":
    fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/"
else:
    fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/GA_params/"


simtime = pars.T_total # 
for seed in seeds:
    if ga == "n":
        if sim_type == "non_bursty":
            name = subject+"_"+state#+"_"+channel
            gpe_prefix = 'GP_gp_'
            stn_prefix = 'ST_gp_'
        elif sim_type == "bursty":
            name = subject+"_"+state+"_gpe_ratio_"+str(gpe_ratio)+"_stn_ratio_"+str(stn_ratio)
            gpe_prefix = 'GP_gp_bursty_'
            stn_prefix = 'ST_gp_bursty_'
    else:
            name = subject+"_"+state#+"_"+channel
            gpe_prefix = 'GP_gp_'
            stn_prefix = 'ST_gp_'

    if ga == "n":
        f_name_gp =  gpe_prefix + str(gpe_rat) + '_stn_' +name+ "-3001-0"+".gdf"
        f_name_stn = stn_prefix + str(gpe_rat) + '_stn_' +name+"-3002-0"+".gdf"
    else:

        f_name_gp = 'GP_gp_' + str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+"_"+str(scale_gpe_inh)+"_"+str(scale_stn_exc)+"_"+str(scale_synaptic)+"_"+str(scale_conn)+"_"+str(scale_delays)+"_"+str(name)+"-3001-0"+".gdf"
        f_name_stn = 'ST_gp_' + str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+"_"+str(scale_gpe_inh)+"_"+str(scale_stn_exc)+"_"+str(scale_synaptic)+"_"+str(scale_conn)+"_"+str(scale_delays)+"_"+str(name)+"-3002-0"+".gdf"


    if ga == "n":
        gpe_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_gp)
        stn_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_stn)
    else:
        gpe_act = np.loadtxt("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_gp)
        stn_act = np.loadtxt("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_stn)

    #lim = int(np.max(stn_act[:,1]))
    lim = simtime

    ind_ts_stn = np.where(stn_act[:,1] <=lim)
    a1_stn,b1_stn = np.histogram(stn_act[ind_ts_stn,1],bins=np.arange(0,simtime,binw))

    ind_ts_gpe = np.where(gpe_act[:,1] <=lim)
    a1_gpe,b1_gpe = np.histogram(gpe_act[ind_ts_gpe,1],bins=np.arange(0,simtime,binw))

    ei_info = dict()

    ei_info["stn_avg_rate"]= a1_stn/(0.1*pars.NSTN) # in Hz
    ei_info["gpe_avg_rate"]= a1_gpe/(0.1*pars.NGPe) # in Hz
    ei_info["stn_spec_entropy"]=[]
    ei_info["gpe_spec_entropy"]=[]
    ei_info["gpe_inhibition"]=[]
    ei_info["stn_excitation"]=[]
    for i,time in enumerate(np.arange(0,simtime-binw,binw)):
        #se_stn = sim_anal.spec_entropy(stn_act,binw,time_range=[time,time+binw],freq_range=[10.,40.],Fs=2045.)
        se_stn = sim_anal.percentage_power_in_band(stn_act,binw,time_range=[time,time+binw],freq_range=[10.,40.],Fs=2045.)
        #se_gpe = sim_anal.spec_entropy(gpe_act,binw,time_range=[time,time+binw],freq_range=[10.,40.],Fs=2045.)
        se_gpe = sim_anal.percentage_power_in_band(gpe_act,binw,time_range=[time,time+binw],freq_range=[10.,40.],Fs=2045.)
       
        ei_info["stn_spec_entropy"].append(se_stn)
        ei_info["gpe_spec_entropy"].append(se_gpe)

        gpe_inh = ei_info["gpe_avg_rate"][i] * pars.epsilon_gpe_gpe * pars.NGPe * pars.J_gpe_gpe * pars.tau_syn_in
        stn_exc = ei_info["stn_avg_rate"][i] * pars.epsilon_stn_gpe * pars.NSTN * pars.J_stn_gpe * pars.tau_syn_ex

        ei_info["gpe_inhibition"].append(gpe_inh)
        ei_info["stn_excitation"].append(stn_exc)


    if os.path.isdir(data_target_dir+str(seed)) == False:
        os.mkdir(data_target_dir+str(seed))
    pickle.dump(ei_info,open(data_target_dir+str(seed)+"/"+"EI_info_"+subject+"_"+state+"_"+sim_type+".pickle","wb"))


    

    


    
    



