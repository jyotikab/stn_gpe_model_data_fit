
import matplotlib as mpl
#mpl.use('Agg')

import pylab as pl
import numpy as np
import matplotlib.cm as cm
from pylab import *
# Since its a sparse matrix
import scipy.sparse.linalg as sp
import itertools
import marshal
import pickle
# Plotting the nullclines, how they vary with lambda-CTX
import logging
import gc
from scipy.integrate import odeint
import sys
import glob
from operator import attrgetter
import os
import copy
import pdb
from deap import base, creator
from deap import tools
import deap
import random
import GA_params as GA_par
#import scipy.stats as stats



sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/simulations/")

import main_simulation_ga as main_sim_ga

sys.path.append("/home/bahuguna/Work/Data_Alex/stn_gpe_model/stn_gpe_model_data_fit/scripts/common/")
import params_d_ssbn as params_d
sys.path.append("/home/bahuguna/Work/Data_Alex/scripts/")
import analyze as anal

data_target_dir = "/home/bahuguna/Work/Data_Alex/target_data/"

subject_data = pickle.load(open(data_target_dir+"Subject_data.pickle","rb"))

STN_electrodes = ["LS"+str(i)  for i in np.arange(1,5)] +["L"+str(i)  for i in np.arange(1,5)]+["R"+str(i)  for i in np.arange(1,5)]+["RS"+str(i)  for i in np.arange(1,5)]

piece = pickle.load(open(data_target_dir+"piece_wise_rate.pickle","rb"))


# Check if any element of list1 is > any element of list2. Used to check if any of the stn connections are stronger than cortical connections
def compVals(val,list1):
	for y in list1:
		if np.abs(val) >= np.abs(y):
			return True
	return False

def fitness_function(individual):
    #sols = [d1ta,d2ta,fsita,fsiti,tata,tati,tastn,tita,titi,tistn,stnta,stnti,tid2,tad2,d1ti,d2ti,jc1,jc2,jfsictx,jstnctx,d1d1,d1d2,d2d1,d2d2,d2fsi,d1fsi,gpid1,gpiti,gpita,gpistn]		
    sols = list(individual)		
    poi_rate_bkg_gpe = GA_par.bkg_gpe[sols[0]]
    poi_rate_bkg_stn = GA_par.bkg_stn[sols[1]]
    scale_gpe_inh = GA_par.scale_gpe_inh[sols[2]]
    scale_stn_exc = GA_par.scale_stn_exc[sols[3]]
    scale_synaptic = GA_par.scale_synaptic[sols[4]]
    scale_conn = GA_par.scale_conn[sols[5]]
    scale_delays = GA_par.scale_delays[sols[6]]
    sim_type = GA_par.sim_type[sols[7]]
    gpe_ratio = GA_par.gpe_ratio[sols[8]]
    stn_ratio = GA_par.stn_ratio[sols[9]]


    err = dict()

    for st in ["OFF","ON"]:
            # Run the simulation - OFF state
            params = dict()
            err[st] = dict()
            if st == "OFF":
                    params["stn_inp"] = stn_ip_off
            elif st == "ON":
                    params["stn_inp"] = stn_ip_on
            params["stn_bck_rate"] = poi_rate_bkg_stn
            params["gpe_inp"] = poi_rate_bkg_gpe
            params["simtime"] = 5000
            params["seed"] = seed1
            params["scale_gpe_inh"] = scale_gpe_inh
            params["scale_stn_exc"] = scale_stn_exc
            params["scale_synaptic"] = scale_synaptic
            params["scale_conn"] = scale_conn
            params["scale_delays"] = scale_delays
            params["sim_type"] = sim_type
            params["gpe_ratio"] = gpe_ratio
            params["stn_ratio"] = stn_ratio
            params["name"] = subject1+"_"+st 
            params["path"] = path
            gpe_act, stn_act = main_sim_ga.runSim(params)	

            a_stn,b_stn = np.histogram(stn_act[:,0],bins = np.arange(500.,simtime,1))
            freq,fft = anal.calc_fft(a_stn)
            fft = fft[:int(len(freq)/2)]
            freq = freq[:int(len(freq)/2)]

            #ind_freq1 = np.where(freq<=90)
            #fft = fft[ind_freq1]
            #freq = freq[ind_freq1]
            err[st]["left"] = np.sum((stn_fft[st]["left"][:-1] - fft)**2)
            err[st]["right"] = np.sum((stn_fft[st]["right"][:-1] - fft)**2)

    #print err
    print( "Error ",err)

    if (err["ON"]["left"] <= 0.1 and err["OFF"]["left"] <=0.001) or (err["ON"]["right"] <= 0.1 and err["OFF"]["right"] <=0.001):
            if list(individual) not in ans["params"]:
                    ans["params"].append(params)
                    ans["errors"].append(err)	
                    print("solution found !")
                    print(params)			
                    #ans["stats"] = statsVsIters
                    pickle.dump(ans,open(path+"Combinations_"+str(seed1)+".pickle","wb"))
    #return Errors,
    err_num = err["ON"]["left"]+ err["OFF"]["left"]+err["ON"]["right"]+ err["OFF"]["right"]
    return err_num,


def paramSearch(subject,seed,anti=0):

	if os.path.isdir(data_target_dir+str(seed)+"/"+subject) == False:
		os.mkdir(data_target_dir+str(seed)+"/"+subject)

	global simtime 
	simtime = 5000
	ch_ctx = "C3"
	global stn_fft
	stn_fft = dict()
	global subject1 
	subject1 = subject
	#global seed
	#seed = seed
	for state in ["ON","OFF"]:
		orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in STN_electrodes if ch in list(subject_data[subject][state].keys())]
		stn_fft[state] = dict()
		temp_l = []
		temp_r = []
		for i,(ch,ts) in enumerate(orig_ts):
			ts1 = ts['ts'].T[0][500:int(simtime)]
			ts1_filt = anal.remove_band_signal(ts1,50,2048)
			freq,fft = anal.calc_fft(ts1_filt)
			if "L" in ch:
				temp_l.append(fft)
			elif "R" in ch:
				temp_r.append(fft)



		stn_fft[state]["left"] = np.mean(temp_l,axis=0)[:int(len(freq)/2)]
		stn_fft[state]["right"] = np.mean(temp_r,axis=0)[:int(len(freq)/2)]
		stn_fft[state]["freq"] = freq[:int(len(freq)/2)]
		'''	
		# Frequency cutoff at 90Hz)
		ind_freq = np.where(freq<=90)
		stn_fft[state]["left"] = stn_fft[state]["left"][ind_freq]
		stn_fft[state]["right"] = stn_fft[state]["right"][ind_freq]
		stn_fft[state]["freq"] = stn_fft[state]["freq"][ind_freq]
		'''
	global path
	path = data_target_dir+str(seed)+"/"+subject+"/"

	global stn_ip_on
	stn_ip_on = piece[subject]["ON"][ch_ctx]
	global stn_ip_off 
	stn_ip_off = piece[subject]["OFF"][ch_ctx]


	print( "in Paramsearch")

	logging.basicConfig(level=logging.DEBUG)
	# Time trajectories
	leak = -0.1

	global seed1
	seed1 = int(seed)

	np.random.seed(np.random.randint(0,99999999,1)[0])
	#np.random.seed(seed1)
	#numEpochs = 2500
	
	global posInd
	posInd = np.array([6,9,16,17,18,19,29])
	global negInd
	negInd = np.array(list(set(np.arange(0,30))-set(posInd)))
	# Generate 300 uniform distributions for each unknown parameter
	
	global ans
	ans = dict()
	ans["params"] = []
	ans["errors"] = []

	def findUniq(sols,errs):
		Allstrs = [ np.array2string(np.array(seq)) for seq in sols]
		s,uniqueIds = np.unique(Allstrs, return_index=True)
		uniqSols = np.array(sols)[uniqueIds]
		uniqErrs = np.array(errs)[uniqueIds]

		return uniqSols, uniqErrs

	terms = ["bkg_gpe","bkg_stn","scale_gpe_inh","scale_synaptic","scale_conn","scale_delays"]

	def get_ind():
		return np.random.randint(0,10,1)[0] 
	
	creator.create("FitnessMin",base.Fitness, weights=(-1.0,))
	creator.create("Individual",list, fitness=creator.FitnessMin)
	#import pdb
	#pdb.set_trace()

	toolbox = base.Toolbox()
	#toolbox.register("attribute", get_individual )
	toolbox.register("attr_ind",get_ind)

	toolbox.register("individual", tools.initCycle, creator.Individual,
				 (toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind), n=1)

	toolbox.register("population", tools.initRepeat, list, toolbox.individual)


	toolbox.register("mate", tools.cxUniform,indpb=0.1)
	#toolbox.register("mutate", tools.mutGaussian)
	toolbox.register("mutate", tools.mutUniformInt,low=0,up=9,indpb=0.8)
	#toolbox.register("mutate", tools.mutPolynomialBounded,eta=0.5)
	toolbox.register("select", tools.selTournament, tournsize=3)
	#toolbox.register("select", tools.selBest)
	toolbox.register("evaluate", fitness_function)

	#toolbox.decorate("mutate",checkBounds(params["negMax"]*2.5,0,params["posMax"]*1.5))	
	#toolbox.decorate("mate",checkBoundsCrossover(params["negMax"]*2.5,0,params["posMax"]*1.5))	

	pop = toolbox.population(n=20)
	#pop = toolbox.population(n=10)
	#print pop
	CXPB, MUTPB, NGEN = 0.95, 0.8, 100
	# Evaluate the entire population
	fitnesses = map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	stats = tools.Statistics(key=lambda ind: ind.fitness.values)

	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	#ans["gaParams"] = ga.gaParams

	global statsVsIters
	statsVsIters = []

	#hof = deap.tools.HallOfFame(10)
	global hof
	hof = tools.ParetoFront()

	#pop = [ x for x in pop if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ]	
	fitnesses = map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	for g in np.arange(NGEN):
		# Select the next generation individuals
		offspring = toolbox.select(pop, len(pop))
	# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
		print(offspring)  
	# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

			for mutant in offspring:
				if random.random() < MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values
		
		# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			if hof is not None:
				hof.update(offspring)	

			# The population is entirely replaced by the offspring
			pop[:] = toolbox.select(pop+offspring+[x for x in hof],10)

	return pop		



	'''
	for g in xrange(NGEN):
		# Select the next generation individuals
		#offspring = toolbox.select(pop, len(pop))
		#offspring = toolbox.select(pop, ga.gaParams["pop_size"])

		#offspring = [ x for x in offspring if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ]
		# Clone the selected individuals
		offspring = map(toolbox.clone, pop)

		# Apply crossover and mutation on the offspring
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
			#toolbox.mate(child1, child2,0.5)
				toolbox.mate(child1, child2) # 0.5 is eta for cxSimulatedBinary
				del child1.fitness.values
				del child2.fitness.values
		
		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant,mu=mutant,sigma=1.0,indpb=ga.gaParams["mut_prob_attr"]*2)
			#toolbox.mutate(mutant,indpb=ga.gaParams["mut_prob_attr"]*2,low=params["negMax"],up=params["posMax"])
			#if g <= 10:
			#	toolbox.mutate(mutant,mu=mutant,sigma=1.0,indpb=ga.gaParams["mut_prob_attr"]*2)
			#else:
			#	toolbox.mutate(mutant,mu=mutant,sigma=statsVsIters[g]['avg'],indpb=ga.gaParams["mut_prob_attr"]*2)
				del mutant.fitness.values
		
		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid] # Everything that is not in inavlid_ind was not touched by mating or mutating process, hence their fitness value has alraedy been evaluated in previous iteration ans is valid.
		
		print( len(invalid_ind))
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit


		if hof is not None:
			hof.update(offspring)	

		# The population is entirely replaced by the offspring
		#ind_non_inf = [ x for x in offspring if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ]					# For some reason, there are indivuals with inf as fitnesses on introduction of indpb in mutate, dont pass these tpo next generation. But one has to find out why are infinities created in fitness function 
		#ind_non_inf = [ x for x in offspring if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ]					# For some reason, there are indivuals with inf as fitnesses on introduction of indpb in mutate, dont pass these tpo next generation. But one has to find out why are infinities created in fitness function 
		#pop[:] = [ x for x in offspring if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ]
		#pop[:] = toolbox.select(pop + offspring + hof ,ga.gaParams["pop_size"])
		rand_inds = toolbox.population(ga.gaParams["new_bld"])
		fitnesses_rand = map(toolbox.evaluate, rand_inds)
		for ind, fit in zip(rand_inds, fitnesses_rand):
			ind.fitness.values = fit

		pop[:] = toolbox.select(pop + offspring + [x for x in hof] + rand_inds ,ga.gaParams["pop_size"])
		#pop[:] = toolbox.select(pop + offspring + [x for x in hof] + rand_inds ,10)
		statsVsIters.append(stats.compile(pop))

		#pop[:] = ind_non_inf
		print( "Generation", g)

		ans["stats"] = statsVsIters
		if (g % 10)==0:
			pickle.dump(ans,open(path+"Stats_uniform_dist_new_"+str(seed1)+".pickle","w"))
	#min_err = ga_obj.best_individual()[0]	
	# Take all solutions with 0 error
	for x,y in zip(pop,fitnesses):			
		if list(x) not in ans["sols"] and np.abs(y) <0.001:
			ans["sols"].append(list(x))
			ans["errs"].append(y)
			ans["stats"] = statsVsIters
			pickle.dump(ans,open(path+"Combinations_uniform_dist_new_"+str(seed1)+".pickle","w"))

	'''






