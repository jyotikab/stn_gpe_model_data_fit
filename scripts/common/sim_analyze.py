
import numpy as np
import itertools
import sys
home_directory = ""
sys.path.append(home_directory+'/common/')
#import analyze_data_ssbn as adata
#reload(adata)
import sys
from numpy import *
#reload(adata)
import pdb
import pylab as pl
import matplotlib.cm as cm
import os
import pickle




def comp_mean_rate(act,total_time,numNeurons,binsize,time_range=[]):
	evs = act[:,0]
	ts = act[:,1]
	binsize = binsize
	if time_range!=[]:
	    idx = (ts>time_range[0]) & (ts<=time_range[1])
	    spikes = ts[idx]
	if time_range==[]:
	  total_time =total_time 
	else:
	  total_time = time_range[1] - time_range[0]
	mean_rate = 1.*len(spikes)/(numNeurons*total_time*1e-3)
	return mean_rate
	#psth,xx = np.histogram(ts,bins = np.arange(0,total_time,binsize))
	#return np.mean(psth/((binsize/1000.)*numNeurons))

def psd(act,total_time, bin_w = 5.,time_range = []):
	evs = act[:,0]
	ts = act[:,1]

	if time_range!=[]:
	    idx = (ts>time_range[0]) & (ts<=time_range[1])
	    spikes = ts[idx]
	if time_range==[]:
	  total_time =total_time 
	else:
	  total_time = time_range[1] - time_range[0]

	if len(spikes) == 0:
	  print( 'psd: spike array is empty')
	  return np.nan, np.nan, np.nan, np.nan
	  
	ids = np.unique(evs)
	nr_neurons = len(ids)
	#psd, max_value, freq,h = misc2.psd_sp(spikes[:,1],nr_bins,nr_neurons)
	bins = np.arange(time_range[0],time_range[1],bin_w)
	a,b = np.histogram(spikes, bins)
	ff = abs(np.fft.fft(a- np.mean(a)))**2
	Fs = 1./(bin_w*0.001)
	freq2 = np.fft.fftfreq(len(bins))[0:len(bins/2)+1]
	freq = np.linspace(0,Fs/2,len(ff)/2+1)
	px = ff[0:len(ff)/2+1]
	max_px = np.max(px[1:])
	idx = px == max_px
	corr_freq = freq[pl.find(idx)]
	new_px = px
	max_pow = new_px[pl.find(idx)]
	return new_px,freq, freq2, corr_freq[0], max_pow


def spec_entropy(act,total_time,bin_w = 5.,time_range=[],freq_range = []):
	'''Function to calculate the spectral entropy'''
	power,freq,dummy,dummy,dummy = psd(bin_w = bin_w,time_range = time_range,act=act,total_time=total_time)
	if freq_range != []:
	  power = power[(freq>freq_range[0]) & (freq < freq_range[1])]
	  freq = freq[(freq>freq_range[0]) & (freq < freq_range[1])]	
	k = len(freq)
	power = power/sum(power)
	sum_power = 0
	for ii in range(k):
	  sum_power += (power[ii]*np.log(power[ii]))
	spec_ent = -(sum_power/np.log(k))  
	return spec_ent








def plot_raster(acts,ax,labels,time_range,binw):

    # Assuming that NEST assigns contigous ids
    colors = ['darkorange','steelblue','darkslategrey','sienna']
    for i,ac in enumerate(acts):
        ind_t = np.where(np.logical_and(ac[:,1]>=time_range[0],ac[:,1]<=time_range[1])==True)[0]
        offset = np.min(ac[ind_t,0])
        bins = np.arange(time_range[0],time_range[1],binw)
        a,b = np.histogram(ac[ind_t,1],bins=bins)
        ax.plot(ac[ind_t,1],ac[ind_t,0],'.',color=colors[i],label=labels[i])
        ax.plot(b[:-1],a+offset,'k-',linewidth=4.0)
    ax.legend(prop={'size':10,'weight':'bold'})

   



    
def plot_instantaneous_rate_comparison(orig_ts,stn_act,bin_w,ax):
   
    cmap1 = cm.get_cmap('magma_r',len(orig_ts)+4)
    colors = [ cmap1(i) for i in np.arange(len(orig_ts)+4) ]
    bins = np.arange(0,np.max(stn_act[:,1]),bin_w)

    a,b = np.histogram(stn_act[:,1],bins=bins)

    ax.plot(b[:-1],a,'-',color='steelblue',linewidth=4.0,label='simulation-STN')
    fs = 2045
    for i,(ch,st_ts) in enumerate(orig_ts):
        print(i)
        lim = int(np.max(stn_act[:,1]))
        mean_ts = [ np.nanmean(st_ts['ts'][i:i+bin_w]) for i in np.arange(0,len(st_ts['ts'][:lim]),bin_w) ]
        ax.plot(np.arange(0,len(st_ts['ts'][:lim]),bin_w),mean_ts,'-',color=colors[i+1],linewidth=4.0,label=ch)

    ax.legend(prop={'size':10,'weight':'bold'})


def plot_input(ts_input,ax,ch,lim):
    
    ax.plot(ts_input['orig_ts'][:lim],'-',color='k',linewidth=4.0,label=ch)
    pwr = ts_input['piece_wise_rate']
    ind = np.where(pwr[0]<=lim)[0]
    ax.plot(pwr[0][ind],pwr[1][ind],'-',color='teal',linewidth=4.0,label='piece-wise rate')

    ax.legend(prop={'size':10,'weight':'bold'})





