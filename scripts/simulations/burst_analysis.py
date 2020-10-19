import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import pylab as pl
import pdb
import more_itertools as mit
import matplotlib.cm as cm
import scipy.stats as sp_st
import pandas as pd
import seaborn as sns
import glob
from matplotlib.lines import Line2D

data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/GA_params/" 

files = glob.glob(data_target_dir+"burst_info_*.pickle")
fig_target_dir = "/home/bahuguna/Work/Data_Alex/figs/"
subjects = [ x.split('.')[0].split('_')[-1]  for x in files]
states = ["OFF","ON"]

burst_info_all_df = pd.DataFrame(columns=["subject","state","nuclei","data_type","burst_length","burst_amplitude","model_no"])
temp_dict = dict()
for k in burst_info_all_df.keys():
    temp_dict[k] = []


for i,fi in enumerate(files):
    burst_info = pickle.load(open(fi,"rb"))

    for st in states:
        bi_l_c = burst_info[st]['len_burst_ctx']
        bi_a_c = burst_info[st]['amp_ctx']
        bi_l_s = burst_info[st]['len_burst_stn']
        bi_a_s = burst_info[st]['amp_stn']

        for nuc in ["CTX","STN"]:
            if nuc == "CTX":
                bi_l = bi_l_c
                bi_a = bi_a_c
            elif nuc == "STN":
                bi_l = bi_l_s
                bi_a = bi_a_s

            temp_dict["burst_length"].append(bi_l)
            temp_dict["burst_amplitude"].append(bi_a)
            temp_dict["subject"].append([subjects[i] for k in np.arange(len(bi_l))])
            temp_dict["state"].append([st for k in np.arange(len(bi_l))])
            temp_dict["nuclei"].append([nuc for k in np.arange(len(bi_l))])
            temp_dict["data_type"].append(["data" for k in np.arange(len(bi_l))])
            temp_dict["model_no"].append([0 for k in np.arange(len(bi_l))])


        model_nums = [ x for x in burst_info[st].keys() if str != type(x) ]
        for mn in model_nums:
            bi_l_s = burst_info[st][mn]['len_burst_sim']
            bi_a_s = burst_info[st][mn]['amp_sim']
            nuc = "STN"

            temp_dict["burst_length"].append(bi_l_s)
            temp_dict["burst_amplitude"].append(bi_a_s)
            temp_dict["subject"].append([subjects[i] for k in np.arange(len(bi_l_s))])
            temp_dict["state"].append([st for k in np.arange(len(bi_l_s))])
            temp_dict["nuclei"].append([nuc for k in np.arange(len(bi_l_s))])
            temp_dict["data_type"].append(["simulation" for k in np.arange(len(bi_l_s))])
            temp_dict["model_no"].append([mn for k in np.arange(len(bi_l_s))])

           

for k in burst_info_all_df.keys():
    burst_info_all_df[k] = np.hstack(temp_dict[k])


burst_info_all_df.to_csv(data_target_dir+"burst_info_all.csv")

# Data
bi_data = burst_info_all_df.loc[burst_info_all_df["data_type"]=="data"]


g1_bl = sns.catplot(x="subject",y="burst_length",hue="state",col="nuclei",data=bi_data,kind='box')
g1_bl.fig.savefig(fig_target_dir+"Burst_length_comparison_data.png")

g1_ba = sns.catplot(x="subject",y="burst_amplitude",hue="state",col="nuclei",data=bi_data,kind='box',sharey=False)
g1_ba.fig.savefig(fig_target_dir+"Burst_amplitude_comparison_data.png")


cmap1 = cm.get_cmap('nipy_spectral')
#colors = [cmap1(x)  for x in np.arange(len(subjects)+1)]
colors = ['darkorange','grey','firebrick','forestgreen','dodgerblue','darkorchid']

fig = pl.figure(figsize=(18,12))
subhands = [fig.add_subplot(1,2,i+1) for i in np.arange(2)]
for j,nuc in enumerate(["CTX","STN"]):
    for i,sub in enumerate(subjects):
        for st in states:
            bi_temp = bi_data.loc[(bi_data["subject"]==sub)&(bi_data["nuclei"]==nuc)&(bi_data["state"]==st)]
            if st == "ON":
                scatter_kws = {'marker':'o'}
                line_kws ={'lw':4.5,'ls':'solid'}
            elif st == 'OFF':
                scatter_kws = {'marker':'s'}
                line_kws ={'lw':4.5,'ls':'dashed'}
            if st == "OFF":
                sns.regplot(x="burst_length",y="burst_amplitude",ax=subhands[j],color=colors[i],scatter_kws=scatter_kws,line_kws=line_kws,label=sub,data=bi_temp)
            else:
                sns.regplot(x="burst_length",y="burst_amplitude",ax=subhands[j],color=colors[i],scatter_kws=scatter_kws,line_kws=line_kws,data=bi_temp)
    subhands[j].set_title(nuc,fontsize=15,fontweight='bold')
    subhands[j].legend(prop={'size':12,'weight':'bold'})


lines = [Line2D([0], [0], color='darkorange', linewidth=1.5,linestyle=c) for c in ['dashed','solid'] ]

fig.legend(lines,["OFF","ON"],bbox_to_anchor=(0.53,0.94),prop={'size':12,'weight':'bold'})


fig.savefig(fig_target_dir+"Burst_length_amplitude_pooled_data.png")
        
        
         




# Simulation
bi_sim = burst_info_all_df.loc[burst_info_all_df["data_type"]=="simulation"]
g1_bl = sns.catplot(x="subject",y="burst_length",hue="state",col="nuclei",data=bi_sim,kind='box')
g1_bl.fig.savefig(fig_target_dir+"Burst_length_comparison_simulation.png")

g1_ba = sns.catplot(x="subject",y="burst_amplitude",hue="state",col="nuclei",data=bi_sim,kind='box',sharey=False)
g1_ba.fig.savefig(fig_target_dir+"Burst_amplitude_comparison_simulation.png")

fig = pl.figure(figsize=(12,12))
subhands = [fig.add_subplot(1,1,i+1) for i in np.arange(1)]
for j,nuc in enumerate(["STN"]):
    for i,sub in enumerate(subjects):
        for st in states:
            bi_temp = bi_sim.loc[(bi_sim["subject"]==sub)&(bi_sim["nuclei"]==nuc)&(bi_sim["state"]==st)]
            if st == "ON":
                scatter_kws = {'marker':'o'}
                line_kws ={'lw':4.5,'ls':'solid'}
            elif st == 'OFF':
                scatter_kws = {'marker':'s'}
                line_kws ={'lw':4.5,'ls':'dashed'}
            if st == "OFF":
                sns.regplot(x="burst_length",y="burst_amplitude",ax=subhands[j],color=colors[i],scatter_kws=scatter_kws,line_kws=line_kws,label=sub,data=bi_temp)
            else:
                sns.regplot(x="burst_length",y="burst_amplitude",ax=subhands[j],color=colors[i],scatter_kws=scatter_kws,line_kws=line_kws,data=bi_temp)
    subhands[j].set_title(nuc,fontsize=15,fontweight='bold')
    subhands[j].legend(prop={'size':12,'weight':'bold'})


lines = [Line2D([0], [0], color='darkorange', linewidth=1.5,linestyle=c) for c in ['dashed','solid'] ]

fig.legend(lines,["OFF","ON"],bbox_to_anchor=(0.3,0.94),prop={'size':12,'weight':'bold'})


fig.savefig(fig_target_dir+"Burst_length_amplitude_pooled_simulation.png")


# Model wise comparison
for i,sub in enumerate(subjects):
    bi_sim_sub = bi_sim.loc[bi_sim["subject"]==sub]

    if os.path.exists(fig_target_dir+"/"+sub) == False:
        os.mkdir(fig_target_dir+"/"+sub)

    g1 = sns.catplot(x="model_no",y="burst_length",hue="state",data=bi_sim_sub,kind='box')
    g1.fig.savefig(fig_target_dir+"/"+sub+"/Burst_length_simulation_comparison.png")


    g2 = sns.catplot(x="model_no",y="burst_amplitude",hue="state",data=bi_sim_sub,kind='box')
    g2.fig.savefig(fig_target_dir+"/"+sub+"/Burst_amplitude_simulation_comparison.png")

