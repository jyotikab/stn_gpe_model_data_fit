
import numpy as np
import itertools
import shutil
import os
import pickle
import sys

sim_name = "PD_GA_uniform_dist_new"
storage_home =  os.getcwd()

path1 = storage_home+'/scripts/' # Path to store the results
path2 = storage_home+'/Dist' # Path to jdf files
path3 = storage_home+'/GA/'
path5 = storage_home+'/output/' # Path for output 

for x in np.arange(10): # Run 10 scripts in parallel with a different random seed everytime 

    seed = np.random.randint(0,999999999,1)[0]

    postfix =""
    filename = path1 + '%s%d_%s.py'%(sim_name,seed,postfix)
    fh = open(filename,'w')
    fh.write('import sys\n')
    #fh.write("sys.path.insert(0,'')\n") 
    fh.write("sys.path.insert(0,'%s')\n"%(storage_home))
    fh.write('import fitness_func_deap_uniform_dist as fit_fn\n')
    fh.write('delay = 1.0\n')
    fh.write('fit_fn.paramSearch(delay,%d)\n'%(seed))
    fh.close()
    print(sim_name)

    os.system("python "+filename+" &")

