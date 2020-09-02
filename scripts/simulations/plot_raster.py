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




data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/" 

piece = pickle.load(open(data_target_dir+"piece_wise_rate.pickle","rb"))
subject_data = pickle.load(open(data_target_dir+"Subject_data.pickle","rb"))
STN_electrodes = ["LS"+str(i)  for i in np.arange(1,5)] +["L"+str(i)  for i in np.arange(1,5)]+["R"+str(i)  for i in np.arange(1,5)]+["RS"+str(i)  for i in np.arange(1,5)]
binw = 20
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

pars = params_d.Parameters()

#def plot_subject_wise_simulation_data_comparison(subject = 'tourraine',state = 'ON',channel = 'C3',seeds = [234]):

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


    ind_wup_gpe = np.where(gpe_act[:,1]>pars.T_wup)[0]
    ind_wup_stn = np.where(stn_act[:,1]>pars.T_wup)[0]

    gpe_act1 = gpe_act[ind_wup_gpe,:]
    stn_act1 = stn_act[ind_wup_stn,:]


    lim = int(np.max(stn_act[:,1]))
    orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in STN_electrodes if ch in list(subject_data[subject][state].keys())]


    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+state+" - input: "+channel,fontsize=15,fontweight='bold')
    t1 = fig.add_subplot(311)
    t2 = fig.add_subplot(312)
    t3 = fig.add_subplot(313)

    #raster_lim = [39000,42000]
    raster_lim = [5000,6000]
    #raster_lim = [1000,2000]
    sim_anal.plot_input(piece[subject][state][channel],t1,channel,lim)
    sim_anal.plot_raster([gpe_act1,stn_act1],t2,["GPe","STN"],raster_lim,binw)
    sim_anal.plot_instantaneous_rate_comparison(orig_ts,stn_act1,binw,t3)

    t2.set_xlim(raster_lim[0],raster_lim[1])

    if os.path.isdir(fig_dir+subject) == False:
        os.mkdir(fig_dir+subject)

    if sim_type == "non_bursty":
        figname = "Simulation_data_comparison_"+state+"_"+channel+".png"
    elif sim_type == "bursty":
        figname = "Simulation_data_comparison_"+state+"_"+channel+"_bursty"+".png"
    fig.savefig(fig_dir+subject+"/"+figname)





#plot_subject_wise_simulation_data_comparison(subject,state,channel,seeds)
