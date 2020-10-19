import numpy as np


bkg_gpe = np.arange(650,1650,100)
bkg_stn = np.arange(350,1450,100)
scale_gpe_inh = np.round(np.linspace(0.0,0.8,10),1)
scale_stn_exc = np.round(np.linspace(0.0,0.8,10),1)
scale_synaptic = np.round(np.linspace(0.3,2.0,10),1)
scale_conn = np.round(np.linspace(0.3,2.0,10),1)
scale_delays = np.round(np.linspace(0.5,5,10),1)

sim_type = ["bursty" if i %2 == 0 else "non_bursty"  for i in np.arange(0,10) ] 
#np.random.shuffle(sim_type)
gpe_ratio = np.round(np.linspace(0.05,0.95,10),2)
stn_ratio = np.round(np.linspace(0.05,0.95,10),2)

input_stn_delay = np.round(np.linspace(1,11,10),1)
input_gpe_delay = np.round(np.linspace(1,11,10),1)



