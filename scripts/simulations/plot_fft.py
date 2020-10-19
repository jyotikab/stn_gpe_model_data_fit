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


#spec_entropy = pickle.load(open(data_target_dir+"spectral_entropies.pickle","rb"))
#ctx_elec1,stn_elec1 = spec_entropy[subject]["CTX_fit"], spec_entropy[subject]["STN_fit"]
#channel = ctx_elec1


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
        #f_name_gp =  gpe_prefix + str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+"_"+str(name)+"-3001-0"+".gdf"
        #f_name_stn = stn_prefix + str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+"_"+str(name)+"-3002-0"+".gdf"
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
    orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in list(subject_data[subject][state].keys()) if ch in STN_electrodes or "dipole" in ch]
    #orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in list(subject_data[subject][state].keys()) if "dipole" in ch ]

    #stn_elec_mask = [ se in list(subject_data[subject][state].keys())   for se in stn_elec1]
    
    #stn_elec_final = np.array(stn_elec1)[stn_elec_mask][0]
    stn_elec_final = electrode 

    fig = pl.figure(figsize=(20,12))
    fig.suptitle(subject+" : "+state+" - input: "+channel,fontsize=15,fontweight='bold')

    t1 = fig.add_subplot(111)
    # Convert the spike time to psth

    cmap1 = cm.get_cmap('magma',len(orig_ts)+2)
    colors = [ cmap1(i) for i in np.arange(len(orig_ts)+2) ]
    left_fft = []
    right_fft = []
    chosen = []
    for i,(ch,ts) in enumerate(orig_ts):
        ts1 = ts['ts'].T[0][:int(simtime)]
        # band pass filter 50 hz, line noise
        ts1_filt = anal.remove_band_signal(ts1,50,2048)
        # Only deal with beta band here
        ts2_filt = anal.calc_filtered_signal(ts1_filt,[10.,35.],2048)
        anal.draw_fft(ts=ts2_filt,ax=t1,tit=ch,color=colors[i],normed=True,lw=1)
        if "L" in ch:
            left_fft.append(ts2_filt)
        elif "R" in ch:
            right_fft.append(ts2_filt)
       

        #if ch == electrode:
        if ch == stn_elec_final:
            chosen.append(ts2_filt)
        
    left_sig = np.nanmean(left_fft,axis=0)
    right_sig = np.nanmean(right_fft,axis=0)

    anal.draw_fft(ts=left_sig,ax=t1,tit="Left-STN",color=colors[1],normed=True,lw=1.0)
    anal.draw_fft(ts=chosen[0],ax=t1,tit=electrode,color=colors[1],normed=True,lw=4.0)
    #anal.draw_fft(ts=right_sig,ax=t1,tit="Right-STN",color=colors[5],normed=True,lw=4.0)



    ind_ts = np.where(stn_act[:,1] <=lim)
    a1,b1 = np.histogram(stn_act[ind_ts,1],bins=np.arange(0,simtime,binw))
    a1_filt = anal.calc_filtered_signal(a1,[10.,35.],fs=1)
    anal.draw_fft(a1_filt,t1,"Simulation - STN",'steelblue')   

    t1.legend(prop={'size':10,'weight':'bold'})
    
    #if sim_type == "non_bursty":
    #    figname = "Simulation_data_fft_comparison_"+state+"_"+channel+".png"
    #elif sim_type == "bursty":
    #    figname = "Simulation_data_fft_comparison_"+state+"_"+channel+"_bursty.png"
    figname = "Simulation_data_fft_comparison_"+state+"_"+channel+ str(gpe_rat) + '_stn_' + str(stn_rat)+"_"+sim_type+"_"+str(gpe_ratio)+"_"+str(stn_ratio)+".png"

    fig.savefig(fig_dir+subject+"/"+index+"/"+figname)


