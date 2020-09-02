
import numpy as np
import itertools
import shutil
import os
import pickle
import sys
import fitness_func_deap as fit_fn

subject = sys.argv[1]
seed = sys.argv[2]

#for x in np.arange(10): # Run 10 scripts in parallel with a different random seed everytime 

#    seed = np.random.randint(0,999999999,1)[0]


fit_fn.paramSearch(subject,seed)

'''
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
'''
