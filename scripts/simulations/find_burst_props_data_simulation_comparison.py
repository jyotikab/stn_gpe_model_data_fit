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

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/GA_scripts/")
import GA_params as GA_par
#import scipy.stats as stats

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/simulations/")

import main_simulation_ga_check as main_sim_ga_check

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import params_d_ssbn as params_d
import sim_analyze as sim_anal  
pars = params_d.Parameters()

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


# Calculate the burst information from the subject's data
simtime = pars.T_total

burst_info_data_simulation = dict()

for st in ["OFF","ON"]:

    if st not in burst_info_data_simulation.keys():
        burst_info_data_simulation[st] = dict()

    ctx_channel = subject_data[subject][st][channel]['ts'][:int(simtime)]
    for stn_el in stn_elec1:
        if stn_el in subject_data[subject][st].keys():
            stn_channel = subject_data[subject][st][stn_el]['ts'].T[0][:int(simtime)]
            break

    ctx_filt = anal.calc_filtered_signal(ctx_channel,[10.,35.],2048)
    stn_filt =  anal.calc_filtered_signal(stn_channel,[10.,35.],2048)

    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+st+" - input: "+channel,fontsize=15,fontweight='bold')
    t1 = fig.add_subplot(211)
    t2 = fig.add_subplot(212)
    #t3 = fig.add_subplot(213)
    
    Sxx_stn,f_stn,t_stn,ind_thresh_stn = anal.draw_spectogram_and_mark_burst(stn_filt,t1,stn_el,thresh=99)
    Sxx_ctx,f_ctx,t_ctx,ind_thresh_ctx = anal.draw_spectogram_and_mark_burst(ctx_filt,t2,channel,thresh=99)

    # calculate burst length
    len_burst_stn,amp_stn = anal.calc_burst_length_and_amplitude(ind_thresh_stn[1],ind_thresh_stn[0],Sxx_stn)
    
    burst_info_data_simulation[st]["len_burst_stn"] = len_burst_stn
    burst_info_data_simulation[st]["amp_stn"] = amp_stn

    len_burst_ctx,amp_ctx = anal.calc_burst_length_and_amplitude(ind_thresh_ctx[1],ind_thresh_ctx[0],Sxx_ctx)
    
    burst_info_data_simulation[st]["len_burst_ctx"] = len_burst_ctx
    burst_info_data_simulation[st]["amp_ctx"] = amp_ctx



    '''
    burst_stn_inds = np.where(np.array([  list(group) for group in mit.consecutive_groups(ind_thresh_stn[1])])>=3)

    ind1 = [i  for i,x in enumerate(burst_stn_inds) if len(x)>=3 ] # burst length at least 0.05*3 = 150ms
    '''


    figname = "Simulation_data_spectogram_comparison_burst_"+st+".png"
    fig.savefig(fig_target_dir+subject+"/"+figname)


cmap1 = cm.get_cmap('Set2')
colors = [ cmap1(i) for i in np.arange(6)]
colors = ["firebrick","dodgerblue"]


fig = pl.figure(figsize=(18,16))
subhands = [ fig.add_subplot(1,2,i+1) for i in np.arange(2)]
for i,nuc in enumerate(["STN","CTX"]):
    for j,st in enumerate(["OFF","ON"]):
        if nuc == "STN":
            x = burst_info_data_simulation[st]["len_burst_stn"]
            y = burst_info_data_simulation[st]['amp_stn']
            subhands[i].plot(x,y,'o',label=st,color=colors[j],ms=15,alpha=0.7)

        elif nuc == "CTX":
            x = burst_info_data_simulation[st]["len_burst_ctx"]
            y = burst_info_data_simulation[st]['amp_ctx']

            subhands[i].plot(x,y,'o',label=st,color=colors[j],ms=15,alpha=0.7)

        slope, intercept, r_value, p_value, std_err = sp_st.linregress(x, y)
        subhands[i].plot(x,np.array(x)*slope+intercept,'-',lw=3.5,color=colors[j])
        subhands[i].text(0.75*np.max(x),0.9*np.max(y),"(r= "+str(np.round(r_value,2))+", p="+str(np.round(p_value,4))+")",fontsize=15,fontweight='bold',color=colors[j])

    subhands[i].set_title(nuc,fontsize=15,fontweight='bold')
    subhands[i].legend(prop={'size':15,'weight':'bold'})
    subhands[i].set_xlabel("Burst length",fontsize=15,fontweight='bold')
    subhands[i].set_ylabel("Burst amplitude",fontsize=15,fontweight='bold')

fig.suptitle(subject,fontsize=15,fontweight='bold')
fig.savefig(fig_target_dir+subject+"/"+"Burst_length_amplitude_data.png")




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

    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+st+" - input: "+channel,fontsize=15,fontweight='bold')
    t1 = fig.add_subplot(211)
    t2 = fig.add_subplot(212)
    subhands1 = [t1,t2]

    fig1 = pl.figure(figsize=(18,16))
    t11 = fig1.add_subplot(111)

    for k,st in enumerate(["OFF","ON"]):
        name = subject+"_"+st#+"_"+channel
        gpe_prefix = 'GP_gp_'
        stn_prefix = 'ST_gp_'

        f_name_gp = 'GP_gp_' + str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+"_"+str(scale_gpe_inh)+"_"+str(scale_stn_exc)+"_"+str(scale_synaptic)+"_"+str(scale_conn)+"_"+str(scale_delays)+"_"+str(name)+"-3001-0"+".gdf"
        f_name_stn = 'ST_gp_' + str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+"_"+str(scale_gpe_inh)+"_"+str(scale_stn_exc)+"_"+str(scale_synaptic)+"_"+str(scale_conn)+"_"+str(scale_delays)+"_"+str(name)+"-3002-0"+".gdf"

        if os.path.exists("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_gp)== True:
            gpe_act = np.loadtxt("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_gp)
            stn_act = np.loadtxt("/home/bahuguna/Work/Data_Alex/target_data/GA_params/"+f_name_stn)
        else:
            continue

        # Convert the spike time to psth
        ind_ts = np.logical_and(stn_act[:,1]>=500.,stn_act[:,1] <=simtime)
        a1,b1 = np.histogram(stn_act[ind_ts,1],bins=np.arange(0,simtime,1))
        a1_filt = anal.calc_filtered_signal(a1,[10.,35.],fs=1)

        Sxx_act,f_act,t_act,ind_thresh_act = anal.draw_spectogram_and_mark_burst(a1_filt,subhands1[k],st,thresh=99)

        len_burst_act,amp_act = anal.calc_burst_length_and_amplitude(ind_thresh_act[1],ind_thresh_act[0],Sxx_act)
        if i+1 not in burst_info_data_simulation[st].keys():
            burst_info_data_simulation[st][i+1] = dict()
        
        burst_info_data_simulation[st][i+1]["len_burst_sim"] = len_burst_act
        burst_info_data_simulation[st][i+1]["amp_sim"] = amp_act

        x = burst_info_data_simulation[st][i+1]["len_burst_sim"]
        y = burst_info_data_simulation[st][i+1]['amp_sim']

        t11.plot(x,y,'o',label=st,color=colors[k],ms=15,alpha=0.7)
        slope, intercept, r_value, p_value, std_err = sp_st.linregress(x, y)
        t11.plot(x,np.array(x)*slope+intercept,'-',lw=3.5,color=colors[k])
        t11.text(0.75*np.max(x),0.9*np.max(y),"(r= "+str(np.round(r_value,2))+", p="+str(np.round(p_value,4))+")",fontsize=15,fontweight='bold',color=colors[k])

    t11.set_title("Simulation:"+str(i+1),fontsize=15,fontweight='bold')
    t11.legend(prop={'size':15,'weight':'bold'})
    t11.set_xlabel("Burst length",fontsize=15,fontweight='bold')
    t11.set_ylabel("Burst amplitude",fontsize=15,fontweight='bold')
    fig1.savefig(fig_target_dir+subject+"/"+str(i+1)+"/"+"Burst_length_amplitude_simulation.png")

    try:
        fig.savefig(fig_target_dir+subject+"/"+str(i+1)+"/"+"Burst_spectrogram_on_off.png")
    except ValueError:
        continue


pickle.dump(burst_info_data_simulation, open(data_target_dir+"/"+"burst_info_data_sim_"+subject+".pickle","wb"))
