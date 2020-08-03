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

fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/"



data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/" 

subject_data = pickle.load(open(data_target_dir+"Subject_data.pickle","rb"))
STN_electrodes = ["LS"+str(i)  for i in np.arange(1,5)] +["L"+str(i)  for i in np.arange(1,5)]+["R"+str(i)  for i in np.arange(1,5)]+["RS"+str(i)  for i in np.arange(1,5)]
binw = 1
subject = sys.argv[1]
state = sys.argv[2]
channel = sys.argv[3]
gpe_rat = sys.argv[4]
seeds = [234]


simtime = pars.T_total # 
for seed in seeds:
    name = subject+"_"+state#+"_"+channel
    f_name_gp = 'GP_gp_' + str(gpe_rat) + '_stn_' +name+ "-3001-0"+".gdf"
    f_name_stn = 'ST_gp_' + str(gpe_rat) + '_stn_' +name+"-3002-0"+".gdf"
    gpe_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_gp)
    stn_act = np.loadtxt(pars.data_path+str(seed)+"/"+f_name_stn)

    #lim = int(np.max(stn_act[:,1]))
    lim = simtime
    orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in STN_electrodes if ch in list(subject_data[subject][state].keys())]

    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+state+" - input: "+channel,fontsize=15,fontweight='bold')

    t1 = fig.add_subplot(111)
    # Convert the spike time to psth
    ind_ts = np.where(stn_act[:,1] <=lim)
    a1,b1 = np.histogram(stn_act[ind_ts,1],bins=np.arange(0,simtime,binw))

    anal.draw_fft(a1,t1,"Simulation - STN",'steelblue')   

    cmap1 = cm.get_cmap('magma_r',len(orig_ts)+2)
    colors = [ cmap1(i) for i in np.arange(len(orig_ts)+2) ]

    for i,(ch,ts) in enumerate(orig_ts):
        ts1 = ts['ts'].T[0][:int(simtime)]
        anal.draw_fft(ts1,t1,ch,colors[i])

    t1.legend(prop={'size':10,'weight':'bold'})
    
    figname = "Simulation_data_fft_comparison_"+state+"_"+channel+".png"
    fig.savefig(fig_dir+subject+"/"+figname)


