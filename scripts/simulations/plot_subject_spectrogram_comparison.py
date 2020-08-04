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

subject_data = pickle.load(open(data_target_dir+"Subject_data.pickle","rb"))
STN_electrodes = ["LS"+str(i)  for i in np.arange(1,5)] +["L"+str(i)  for i in np.arange(1,5)]+["R"+str(i)  for i in np.arange(1,5)]+["RS"+str(i)  for i in np.arange(1,5)]
binw = 1
subject = sys.argv[1]
state = sys.argv[2]
channel = sys.argv[3]
gpe_rat = sys.argv[4]
sim_type = sys.argv[5]
gpe_ratio = sys.argv[6]
stn_ratio = sys.argv[7]

seeds = [234]


simtime = pars.T_total # 
for seed in seeds:
    if sim_type == "non_bursty":
        name = subject+"_"+state#+"_"+channel
        gpe_prefix = 'GP_gp_'
        stn_prefix = 'ST_gp_'
    elif sim_type == "bursty":
        name = subject+"_"+state+"_gpe_ratio_"+str(gpe_ratio)+"_stn_ratio_"+str(stn_ratio)
        gpe_prefix = 'GP_gp_bursty_'
        stn_prefix = 'ST_gp_bursty_'

    f_name_gp =  gpe_prefix + str(gpe_rat) + '_stn_' +name+ "-3001-0"+".gdf"
    f_name_stn = stn_prefix + str(gpe_rat) + '_stn_' +name+"-3002-0"+".gdf"
    gpe_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_gp)
    stn_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_stn)

    #lim = int(np.max(stn_act[:,1]))
    lim = simtime
    orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in STN_electrodes if ch in list(subject_data[subject][state].keys())]

    # Pick at random two spectrograms
    ind = np.arange(0,len(orig_ts))
    np.random.shuffle(ind)
    ind_plot = ind[:2]

    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+state+" - input: "+channel,fontsize=15,fontweight='bold')
    t1 = fig.add_subplot(311)
    t2 = fig.add_subplot(312)
    t3 = fig.add_subplot(313)

    # Convert the spike time to psth
    ind_ts = np.where(stn_act[:,1] <=lim)
    a1,b1 = np.histogram(stn_act[ind_ts,1],bins=np.arange(0,simtime,binw))
    
    subhands = [t2,t3]
    anal.draw_spectogram(a1,t1,"Simulation - STN")
    for i,i1 in enumerate(ind_plot):
        ch,ts = orig_ts[i1]
        orig_ts1 = ts['ts'].T[0][:int(simtime)]
        
        anal.draw_spectogram(orig_ts1,subhands[i],ch)
    t1.set_xlabel("")
    t2.set_xlabel("")

   
    if os.path.isdir(fig_dir+subject) == False:
        os.mkdir(fig_dir+subject)

    if sim_type == "non_bursty":
        figname = "Simulation_data_spectogram_comparison_"+state+"_"+channel+".png"
    elif sim_type == "bursty":
        figname = "Simulation_data_spectogram_comparison_"+state+"_"+channel+"_bursty.png"

    fig.savefig(fig_dir+subject+"/"+figname)

