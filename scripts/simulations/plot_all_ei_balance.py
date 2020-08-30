import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import pdb
import pylab as pl
import matplotlib.cm as cm
import argparse
sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import sim_analyze as sim_anal  
import params_d_ssbn as params_d
pars = params_d.Parameters()

sys.path.append("/home/bahuguna/Work/Data_Alex/scripts/")
import analyze as anal

fig_dir = "/home/bahuguna/Work/Data_Alex/figs/stn_gpe_model/"
data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/" 


# defined command line options
# this also generates --help and error handling
'''
CLI=argparse.ArgumentParser()
CLI.add_argument(
  "--subjects",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=['tourraine'],  # default if nothing is provided
)
CLI.add_argument(
  "--states",
  nargs="*",
  type=float,  # any type/callable can be used here
  default=['ON','OFF'],
)
CLI.add_argument(
  "--simtype",
  nargs="*",
  type=float,  # any type/callable can be used here
  default=["non_bursty","bursty"],
)
# parse the command line
args = CLI.parse_args()
'''

subjects = sys.argv[1].split(",")
states = sys.argv[2].split(",")
simtypes = sys.argv[3].split(",")

seeds = [234]
fig = pl.figure(figsize=(20,16))
t1 = fig.add_subplot(121)
t2 = fig.add_subplot(122)
subplots = [t1,t2]

markers_subjects = ['s','o','p','d']
colors_osc = ['orangered','darkorange']
colors_non_osc = ['teal','deepskyblue']


#for i,sub in enumerate(args.subjects):
for i,sub in enumerate(subjects):
    #for st in args.states:
    for k,st in enumerate(states):
        #for simtype in args.simtype:
        for j,simtype in enumerate(simtypes):
            ei_info = pickle.load(open(data_target_dir+str(seeds[0])+"/"+"EI_info_"+sub+"_"+st+"_"+simtype+".pickle","rb"))
           
            print(ei_info["stn_spec_entropy"])
            ind_osc = np.where(np.array(ei_info["stn_spec_entropy"])>0.5)
            ind_non_osc = np.where(np.array(ei_info["stn_spec_entropy"])<0.4)
            # Osc
            if len(ind_osc[0]) > 0:
                subplots[j].plot(np.array(ei_info["stn_excitation"])[ind_osc],np.abs(np.array(ei_info["gpe_inhibition"])[ind_osc]),marker=markers_subjects[i],color=colors_osc[k],markersize=5,alpha=0.4,lw=0)
                subplots[j].plot(np.mean(np.array(ei_info["stn_excitation"])[ind_osc]),np.mean(np.abs(np.array(ei_info["gpe_inhibition"])[ind_osc])),marker=markers_subjects[i],color=colors_osc[k],markersize=12,label=sub+"-"+st+"-osc")
            if len(ind_non_osc[0])>0:
                subplots[j].plot(np.array(ei_info["stn_excitation"])[ind_non_osc],np.abs(np.array(ei_info["gpe_inhibition"])[ind_non_osc]),marker=markers_subjects[i],color=colors_non_osc[k],markersize=5,alpha=0.4,lw=0)
                subplots[j].plot(np.mean(np.array(ei_info["stn_excitation"])[ind_non_osc]),np.mean(np.abs(np.array(ei_info["gpe_inhibition"])[ind_non_osc])),marker=markers_subjects[i],color=colors_non_osc[k],markersize=12,label=sub+"-"+st+"-non_osc")

            subplots[j].set_title(simtype,fontsize=15,fontweight='bold')

            subplots[j].legend(prop={'size':10,'weight':'bold'})

            subplots[j].set_xlim(3000,12000)
            subplots[j].set_ylim(94000,104500)
            subplots[j].set_ylabel("GPe inhibition",fontsize=15,fontweight='bold')
            subplots[j].set_xlabel("STN excitation",fontsize=15,fontweight='bold')
            #subplots[j].fill([0,1100,13000,2000,0,0],[8000,8000,100000,100000,100000,8000],color='lightskyblue',alpha=0.1)
            #subplots[j].fill([0,1100,1500,2000,0,0],[8000,8000,11000,15000,15000,8000],color='lightskyblue',alpha=0.1)
            #subplots[j].fill([1100,2000,5000,5000],[8000,15000,15000,8000],color='olive',alpha=0.1)
            #subplots[j].plot((1100,8000),(13000,100000),'lightskyblue' ,'-')

fig.savefig(fig_dir+"EI_balance_all.png")




        
