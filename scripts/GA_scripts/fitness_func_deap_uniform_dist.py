
import matplotlib as mpl
mpl.use('Agg')

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

import paramsearchGA_DopDep_nonlinear_BL_2step as psGA
import copy
from deap import base, creator
from deap import tools
import deap
import random
#import scipy.stats as stats


sys.path.append("/home/j.bahuguna/homology/allParamsFree/params/")

import knownUnknownParams as p1
import GA_params as ga

maxError = np.float64(12.0)

# Check if any element of list1 is > any element of list2. Used to check if any of the stn connections are stronger than cortical connections
def compVals(val,list1):
	for y in list1:
		if np.abs(val) >= np.abs(y):
			return True
	return False

def fitness_function(individual):
	#sols = [d1ta,d2ta,fsita,fsiti,tata,tati,tastn,tita,titi,tistn,stnta,stnti,tid2,tad2,d1ti,d2ti,jc1,jc2,jfsictx,jstnctx,d1d1,d1d2,d2d1,d2d2,d2fsi,d1fsi,gpid1,gpiti,gpita,gpistn]		
	sols = list(individual)		

	path = "/home/j.bahuguna/homology/allParamsFree/output/"

	d1ta = sols[0]
	d2ta = sols[1]
	fsita =sols[2]
	fsiti = sols[3]
	tata = sols[4]
	tati = sols[5]
	tastn = sols[6]
	tita = sols[7]
	titi = sols[8]
	tistn =sols[9]
	stnta = sols[10]
	stnti = sols[11]
	tid2 = sols[12]
	tad2 = sols[13]
	d1ti = sols[14]
	d2ti = sols[15]
	jc1 = sols[16]
	jc2 = sols[17]
	jfsictx = sols[18]
	jstnctx = sols[19]
		
	d1d1 = sols[20]
	d1d2 = sols[21]
	d2d1 = sols[22]
	d2d2 = sols[23]
	d2fsi = sols[24]
	d1fsi = sols[25]
	gpid1 = sols[26]
	gpiti = sols[27]
	gpita = sols[28]
	gpistn = sols[29]

	err1 = 0
	err = np.zeros(16)
	# Do this first to save time

	#if np.abs(d1d1) < np.abs(d2d2) and np.abs(d1d1) < np.abs(d1d2) and np.abs(d2d1) < np.abs(d2d2) and np.abs(d2d1) < np.abs(d1d2): # D2 > D1 > vice versa	
	if np.abs(gpita) < np.abs(gpiti): # D2 > D1 > vice versa	
		err[12] = 0
	else:
		err1 = err1+np.abs(np.abs(gpita)-np.abs(gpiti))
				
	# d2fsi > d1fsi
	if np.abs(d2fsi) > np.abs(d1fsi):
		err[13] = 0
	else:
		#return maxError,
		err1 = err1+np.abs(np.abs(d1fsi)-np.abs(d2fsi))
		#return ,
	if err1 > 0:
		return np.abs(err1+ga.gaParams["maxError"]),
	# Only print combinations that passed the initial test

	#print sols 
	A = np.matrix([[d1d1,d1d2,d1fsi,d1ta,d1ti,0.,0.],[d2d1,d2d2,d2fsi,d2ta,d2ti,0.,0.],[0.,0.,0,fsita,fsiti,0.,0.],[0,tad2,0.,tata,tati,tastn,0],[0.,tid2,0.,tita,titi,tistn,0.],[0.,0.,0.,stnta,stnti,0.,0.],[gpid1,0.,0.,gpita,gpiti,gpistn,0]])
	print( np.round(A,2))
	B = np.matrix([jc1,jc2,jfsictx,0,0,jstnctx,0])
	print( np.round(B,2))
	delay = 1.0	
	#Calculate Rates for SWA and Control
	ipctx = dict()
	ipctx["ip"]=np.zeros((1,2001))	
	#Calculate Rates for SWA and lesion(dopamine depletion)
	Flags = []
	Flags.append("SWA")
	#SWADopDepRates = calcRates(Flags,delay,A,B,False,ipctx,iptau=14.)
	SWADopDepRates = psGA.calcRates(Flags,delay,A,B,False,ipctx,iptau=params["iptau"])

	Flags = []
	Flags.append("Act")
	#ActDopDepRates = calcRates(Flags,delay,A,B,False,ipctx,iptau=14.)
	ActDopDepRates = psGA.calcRates(Flags,delay,A,B,False,ipctx,iptau=params["iptau"])



	if np.round(np.mean(SWADopDepRates['ti'])) >= 19 and np.round(np.mean(SWADopDepRates['ti']))<=60: 	
		err[0] = 0	# 
	else:
		minLim = abs(np.round(np.mean(SWADopDepRates['ti']))-19)/(60.-19.)		# Normalizing the error between 0 and 1, there are smarter ways to do this, will think later  
		maxLim = abs(np.round(np.mean(SWADopDepRates['ti']))-60)/(60.-19.)
		err[0] = np.min([minLim,maxLim])

	if np.round(np.mean(ActDopDepRates['ti'])) >= 7 and np.round(np.mean(ActDopDepRates['ti'])) <=19:		# Mean around 14
		err[1] = 0
	else:
		minLim = abs(np.round(np.mean(ActDopDepRates['ti']))-7)/(19.-7.)  
		maxLim = abs(np.round(np.mean(ActDopDepRates['ti']))-19)/(19.-7.)
		err[1] = np.min([minLim,maxLim])
	
	#if np.mean(SWADopDepRates['ti']) > np.mean(ActDopDepRates['ti']):
	#if np.mean(ActDopDepRates['ta']) > 16 and np.mean(ActDopDepRates['ta']) <=23: # Mean around 19 # 
	if np.round(np.mean(ActDopDepRates['ta'])) >= 7 and np.round(np.mean(ActDopDepRates['ta'])) <=15: # Mean around 19
		err[2] = 0
	else:
		minLim = abs(np.round(np.mean(ActDopDepRates['ta']))-7)/(15.-7.)  
		maxLim = abs(np.round(np.mean(ActDopDepRates['ta']))-15)/(15.-7.)
		err[2] = np.min([minLim,maxLim])
	

	if np.round(np.mean(SWADopDepRates['ta'])) >=1 and np.round(np.mean(SWADopDepRates['ta'])) <=6: # Mean is 12   < np.mean(ActDopDepRates['ta']):
		err[3] = 0
	else:
		minLim = abs(np.round(np.mean(SWADopDepRates['ta']))-1)/(6.-1.)  
		maxLim = abs(np.round(np.mean(SWADopDepRates['ta']))-6)/(6.-1.)
		err[3] = np.min([minLim,maxLim])


	# Refer Fig 5 A,B of Mallet 2008
	GpeSWADopDep = np.mean(SWADopDepRates['ti']+SWADopDepRates['ta'])
	GpeActDopDep = np.mean(ActDopDepRates['ti']+ActDopDepRates['ta'])

	# These conditions can be separated 
	if np.mean(SWADopDepRates['ti']) > np.mean(ActDopDepRates['ti']): 
		err[4] = 0
	else:
		# not sure of this error calculation. Normalized w.r.t to the bigger quantity
		err[4] = (np.mean(ActDopDepRates['ti']) - np.mean(SWADopDepRates['ti']))/np.mean(SWADopDepRates['ti'])


	if np.mean(SWADopDepRates['ta']) < np.mean(ActDopDepRates['ta']):
		err[5] = 0
	else:
		# not sure of this error calculation.
		err[5] = (np.mean(SWADopDepRates['ta']) - np.mean(ActDopDepRates['ta']))/np.mean(ActDopDepRates['ta'])		


	# Check if STn is in phase with ctx (in-phase)
	if SWADopDepRates['stn_ctx'] > 0 and SWADopDepRates['stn_ta'] > 0 and SWADopDepRates['stn_ti'] < 0 :
		err[6] = 0
	else:
		stnctx = 0 if SWADopDepRates['stn_ctx'] > 0 else abs(SWADopDepRates['stn_ctx']-0)
		stnta = 0 if SWADopDepRates['stn_ta'] > 0 else abs(SWADopDepRates['stn_ta']-0)
		stnti = 0 if SWADopDepRates['stn_ti'] < 0 else abs(SWADopDepRates['stn_ti']-0)
		err[6] = stnctx+stnta+stnti
	

	#Shaorrt et .al
	if SWADopDepRates['d1_ctx'] > 0 and SWADopDepRates['d2_ctx'] > 0 and ActDopDepRates['d2_ctx'] < 0:
		err[7] = 0
	else:
		d1ctx = 0 if SWADopDepRates['d1_ctx'] > 0 else abs(SWADopDepRates['d1_ctx']-0)
		d2ctx = 0 if SWADopDepRates['d2_ctx'] > 0 else abs(SWADopDepRates['d2_ctx']-0)
		d2ctxA = 0 if ActDopDepRates['d2_ctx'] < 0 else abs(ActDopDepRates['d2_ctx']-0)
		err[7] = d1ctx+d2ctx+d2ctxA

	# Sharott et. al ( Can be separated)
	if np.mean(ActDopDepRates['d2']) >= 0.1 and  np.mean(ActDopDepRates['d2']) <= 15. and np.mean(ActDopDepRates['d1']) >= 0.1 and np.mean(ActDopDepRates['d1']) <= 5. :
		err[8] = 0
	else:
		d2min = abs(np.mean(ActDopDepRates['d2'])-0.1)/(15.-0.1)	
		d2max = abs(np.mean(ActDopDepRates['d2'])-15)/(15.-0.1)	
		d1min = abs(np.mean(ActDopDepRates['d1'])-0.1)/(5.-0.1)	
		d1max = abs(np.mean(ActDopDepRates['d1'])-5)/(5.-0.1)	
		err[8] = np.min([d2min,d2max])+np.min([d1min,d1max])

	if SWADopDepRates['taFF'] > 2 or SWADopDepRates['tiFF'] > 2:
		err[9] = 0 # Commented because no combination was fulfilling this. Moreover, this is no where specified in mallet.
	else:
		taFF = 0 if SWADopDepRates['taFF'] > 2 else abs(SWADopDepRates['taFF']-2)
		tiFF = 0 if SWADopDepRates['tiFF'] > 2 else abs(SWADopDepRates['tiFF']-2)
		err[9] = np.min([taFF,tiFF])
	# Tests from Sharott et. al
	if np.mean(SWADopDepRates['d2']) >= 0.5 and  np.mean(SWADopDepRates['d2']) <= 5. and np.mean(SWADopDepRates['d1']) >= 0.5 and np.mean(SWADopDepRates['d1']) <= 2. :	# Either impose rate condition here to have weights decrease as a emergent or other way round
		err[10] = 0
	else:
		d2min = abs(np.mean(SWADopDepRates['d2'])-0.5)/(5.-0.5)	
		d2max = abs(np.mean(SWADopDepRates['d2'])-5)/(5.-0.5)	
		d1min = abs(np.mean(SWADopDepRates['d1'])-0.5)/(2.-0.5)	
		d1max = abs(np.mean(SWADopDepRates['d1'])-2)/(2.-0.5)	
		err[10] = np.min([d2min,d2max])+np.min([d1min,d1max])
	# Sanity test , all rates are fairly above zero
	if np.mean(SWADopDepRates['d1'][100./params["dt"]:]) > 0.1 and np.mean(SWADopDepRates['d2'][100./p1.params["dt"]:]) > 0.1 and np.mean(SWADopDepRates['fsi'][100./p1.params["dt"]:]) > 0.1 and	np.mean(SWADopDepRates['ta'][100./p1.params["dt"]:]) > 1.0 and np.mean(SWADopDepRates['ti'][100./p1.params["dt"]:]) > 1.0 and np.mean(SWADopDepRates['stn'][100./p1.params["dt"]:]) > 1.0 and np.mean(SWADopDepRates['gpi'][100./p1.params["dt"]:]) > 0.5:
		err[11] = 0
	else:
		err[11] = 1. # Should be improved, all conditions separated and error of say 1 be added for all rates which are zero



	print( "SWA:d1,d2,fsi,ta,ti,stn,gpi",np.mean(SWADopDepRates["d1"]),np.mean(SWADopDepRates["d2"]),np.mean(SWADopDepRates["fsi"]),np.mean(SWADopDepRates["ta"]),np.mean(SWADopDepRates["ti"]),np.mean(SWADopDepRates["stn"]),np.mean(SWADopDepRates["gpi"]))
	print( "Act:d1,d2,fsi,ta,ti,stn,gpi",np.mean(ActDopDepRates["d1"]),np.mean(ActDopDepRates["d2"]),np.mean(ActDopDepRates["fsi"]),np.mean(ActDopDepRates["ta"]),np.mean(ActDopDepRates["ti"]),np.mean(ActDopDepRates["stn"]),np.mean(ActDopDepRates["gpi"]))
	print( "SWA:d1_ctx,d2_ctx,Act:d1_ctx,d2_ctx",SWADopDepRates['d1_ctx'],SWADopDepRates['d2_ctx'],ActDopDepRates['d1_ctx'],ActDopDepRates['d2_ctx'])
	print( "SWA: taFF,tiFF",SWADopDepRates["taFF"],SWADopDepRates["tiFF"])
	Errors = np.sum(err)
	#print err
	print( "Error ",Errors)

	if np.abs(Errors) <= 0.001:
		if list(individual) not in ans["sols"]:
			ans["sols"].append(sols)
			ans["errs"].append(Errors)	
			ans["stats"] = statsVsIters
			pickle.dump(ans,open(path+"Combinations_uniform_dist_new_"+str(seed1)+".pickle","w"))
	return Errors,


def paramSearch(d,seed,anti=0):

	global params
	params = p1.params.copy()
	path = "/home/j.bahuguna/homology/allParamsFree/output/"
	print( "in Paramsearch")

	logging.basicConfig(level=logging.DEBUG)
	# Time trajectories
	leak = -0.1

	global seed1
	seed1 = seed
	np.random.seed(seed1)
	#numEpochs = 2500
	
	global posInd
	posInd = np.array([6,9,16,17,18,19,29])
	global negInd
	negInd = np.array(list(set(np.arange(0,30))-set(posInd)))
	# Generate 300 uniform distributions for each unknown parameter
	
	global ans
	ans = dict()
	ans["sols"] = []
	ans["errs"] = []

	def findUniq(sols,errs):
		Allstrs = [ np.array2string(np.array(seq)) for seq in sols]
		s,uniqueIds = np.unique(Allstrs, return_index=True)
		uniqSols = np.array(sols)[uniqueIds]
		uniqErrs = np.array(errs)[uniqueIds]

		return uniqSols, uniqErrs

	terms = ["d1ta","d2ta","fsita","fsiti","tata","tati","tastn","tita","titi","tistn","stnta","stnti","tid2","tad2","d1ti","d2ti","jc1","jc2","jfsictx","jstnctx",'d1d1','d1d2','d2d1','d2d2','d2fsi','d1fsi','gpid1','gpiti','gpita','gpistn']

	def get_pos_val():
		return np.round(np.random.uniform(0.0,params["posMax"]*1.5,1)[0],1)
	
	def get_neg_val():
		return np.round(np.random.uniform(params["negMax"]*2.5,0.0,1)[0],1)


	def checkBounds(minNeg,zero,maxPos):
		def decorator(func):
			def wrapper(*args,**kargs):
				offspring = func(*args,**kargs)
				for child in offspring:
					for i in xrange(len(child)):
						#if np.random.rand() < 0.2: # Somehow introducing this condition introduces inf in fitnesses and child values ! No idea why !!
							if i in posInd : # Positive values
								#child[i] = np.round(np.random.normal(loc=child[i],scale=0.2,size=1)[0],1) # first try to sample around the previous point
								# Sample from truncated guassian
								if child[i] < 0 or np.abs(child[i]) > np.abs(maxPos):
									child[i] = np.round(np.random.uniform(zero,maxPos,1)[0],1) # if negative, just sampel uniformly between zero and maxPos
							else:
								#child[i] = np.round(np.random.uniform(lower,upper,size=1)[0],1) # first try to sample around the previous point
								#child[i] = np.round(np.random.normal(loc=child[i],scale=0.2,size=1)[0],1)
								if child[i] > 0 or np.abs(child[i]) > np.abs(minNeg):
									child[i] = np.round(np.random.uniform(minNeg,zero,1)[0],1)
				return offspring
			return wrapper
		return decorator	


	def checkBoundsCrossover(minNeg,zero,maxPos):
		def decorator(func):
			def wrapper(*args,**kargs):
				offspring = func(*args,**kargs)
				for child in offspring:
					for i in xrange(len(child)):
						if i in posInd : # Positive values
							if child[i] < 0 or np.abs(child[i]) > np.abs(maxPos):
								child[i] = np.round(np.random.uniform(zero,maxPos,1)[0],1) # if negative, just sampel uniformly between zero and maxPos
						else:
							if child[i] > 0 or np.abs(child[i]) > np.abs(minNeg):
								child[i] = np.round(np.random.uniform(minNeg,zero,1)[0],1)
				return offspring
			return wrapper
		return decorator	

	creator.create("FitnessMin",base.Fitness, weights=(-1.0,))
	creator.create("Individual",list, fitness=creator.FitnessMin)
	#import pdb
	#pdb.set_trace()

	toolbox = base.Toolbox()
	#toolbox.register("attribute", get_individual )
	toolbox.register("attr_pos",get_pos_val)
	toolbox.register("attr_neg",get_neg_val)

	toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_pos,toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_pos, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg,toolbox.attr_pos,toolbox.attr_pos,toolbox.attr_pos,toolbox.attr_pos,toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg,toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_neg, toolbox.attr_pos ), n=1)

	toolbox.register("population", tools.initRepeat, list, toolbox.individual)


	#toolbox.register("mate", tools.cxTwoPoints)
	toolbox.register("mate", tools.cxUniform,indpb=ga.gaParams["cross_prob_attr"])
	toolbox.register("mutate", tools.mutGaussian)
	#toolbox.register("mutate", tools.mutPolynomialBounded,eta=0.5)
	toolbox.register("select", tools.selTournament, tournsize=10)
	#toolbox.register("select", tools.selBest)
	toolbox.register("evaluate", fitness_function)

	toolbox.decorate("mutate",checkBounds(params["negMax"]*2.5,0,params["posMax"]*1.5))	
	toolbox.decorate("mate",checkBoundsCrossover(params["negMax"]*2.5,0,params["posMax"]*1.5))	

	pop = toolbox.population(n=300)
	#pop = toolbox.population(n=10)
	#print pop
	CXPB, MUTPB, NGEN = ga.gaParams["cross_prob"]*1.2, ga.gaParams["mut_prob"]*1.2, ga.gaParams["generations"] 
	# Evaluate the entire population
	fitnesses = map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	stats = tools.Statistics(key=lambda ind: ind.fitness.values)

	stats.register("avg", np.mean)
	stats.register("std", np.std)
	stats.register("min", np.min)
	stats.register("max", np.max)

	ans["gaParams"] = ga.gaParams

	global statsVsIters
	statsVsIters = []
	hof = deap.tools.HallOfFame(ga.gaParams["hof_size"])
	#pop = [ x for x in pop if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ]	
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
		#ind_non_inf = [ x for x in offspring if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ] 				# For some reason, there are indivuals with inf as fitnesses on introduction of indpb in mutate, dont pass these tpo next generation. But one has to find out why are infinities created in fitness function 
		#ind_non_inf = [ x for x in offspring if (np.isinf(x.fitness.values)==False) and (np.inf not in x) and (-np.inf not in x) ] 				# For some reason, there are indivuals with inf as fitnesses on introduction of indpb in mutate, dont pass these tpo next generation. But one has to find out why are infinities created in fitness function 
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








