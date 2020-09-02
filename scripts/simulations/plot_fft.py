import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import pdb
import pylab as pl
import matplotlib.cm as cm


sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import sim_analyze as sim_anal  
import params_d_ssbn as params_d
pars = params_d.Parameters()

sys.path.append("/home/bahuguna/Work/Data_Alex/scripts/")
import analyze as anal




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
ga = sys.argv[8]

seeds = [234]
if ga == "n":
    fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/"
else:
    fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/GA_params/"

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
    if ga == "n":
        gpe_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_gp)
        stn_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_stn)
    else:
        gpe_act = np.loadtxt("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_gp)
        stn_act = np.loadtxt("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_stn)

    #lim = int(np.max(stn_act[:,1]))
    lim = simtime
    orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in STN_electrodes if ch in list(subject_data[subject][state].keys())]

    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+state+" - input: "+channel,fontsize=15,fontweight='bold')

    t1 = fig.add_subplot(111)
    # Convert the spike time to psth

    cmap1 = cm.get_cmap('magma_r',len(orig_ts)+2)
    colors = [ cmap1(i) for i in np.arange(len(orig_ts)+2) ]

    for i,(ch,ts) in enumerate(orig_ts):
        ts1 = ts['ts'].T[0][:int(simtime)]
        # band pass filter 50 hz, line noise
        ts1_filt = anal.remove_band_signal(ts1,50,2048)
        anal.draw_fft(ts1_filt,t1,ch,colors[i])
    ind_ts = np.where(stn_act[:,1] <=lim)
    a1,b1 = np.histogram(stn_act[ind_ts,1],bins=np.arange(0,simtime,binw))

    anal.draw_fft(a1,t1,"Simulation - STN",'steelblue')   

    t1.legend(prop={'size':10,'weight':'bold'})
    
    if sim_type == "non_bursty":
        figname = "Simulation_data_fft_comparison_"+state+"_"+channel+".png"
    elif sim_type == "bursty":
        figname = "Simulation_data_fft_comparison_"+state+"_"+channel+"_bursty.png"

    fig.savefig(fig_dir+subject+"/"+figname)


