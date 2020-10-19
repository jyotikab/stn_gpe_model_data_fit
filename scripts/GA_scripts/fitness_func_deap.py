
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
#STN_electrodes_dipoles = ["LS"+str(i)  for i in np.arange(1,5)] +["L"+str(i)  for i in np.arange(1,5)]+["R"+str(i)  for i in np.arange(1,5)]+["RS"+str(i)  for i in np.arange(1,5)]

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
    input_stn_delay = GA_par.input_stn_delay[sols[10]]
    input_gpe_delay = GA_par.input_gpe_delay[sols[11]]


    #err = dict()
    corr = dict()


    for st in ["OFF","ON"]:
            # Run the simulation - OFF state
            params = dict()
            #err[st] = dict()
            corr[st] = dict()
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
            params["ip_stn_delay"] = input_stn_delay
            params["ip_gpe_delay"] = input_gpe_delay

            params["name"] = subject1+"_"+st 
            params["path"] = path
            gpe_act, stn_act = main_sim_ga.runSim(params)	

            a_stn,b_stn = np.histogram(stn_act[:,0],bins = np.arange(500.,simtime,1))
            a_stn_filt = anal.calc_filtered_signal(a_stn,[10.,35.],1) 
            freq,fft = anal.calc_fft(a_stn_filt)
            ind_freq = np.logical_and(freq>=0,freq<=60)
            fft = fft[ind_freq]
            freq = freq[ind_freq]

            #ind_freq1 = np.where(freq<=90)
            #fft = fft[ind_freq1]
            #freq = freq[ind_freq1]
            #err[st]["left"] = np.sum((stn_fft[st]["left"][:-1] - fft)**2)
            
            #corr[st]["left"] = np.corrcoef(stn_fft[st]["left"][:-1],fft)[0,1]
            corr[st]["left"] = np.corrcoef(stn_fft[st]["chosen"],fft)[0,1]
            corr[st]["ctx_electrode"] = stn_fft[st]["ctx_electrode"] 
            corr[st]["stn_electrode"] = stn_fft[st]["stn_electrode"] 

            #err[st]["left"] = np.sum(np.abs(stn_fft[st]["left"][:-1] - fft)) # L1 norm
            #err[st]["right"] = np.sum((stn_fft[st]["right"][:-1] - fft)**2)
            #err[st]["right"] = np.sum(np.abs(stn_fft[st]["right"][:-1] - fft))

    #print err
    #print( "Error ",err)
    print( "Correlation ",corr)


    if "tourraine" in params["name"]: # Difficult subject
        #if (err["ON"]["left"] < 0.002 and err["OFF"]["left"] <0.002): #or (err["ON"]["right"] < 0.002 and err["OFF"]["right"] <0.002): # Only for C3, do this for C4
        if (corr["ON"]["left"] >= 0.5 and corr["OFF"]["left"] >=0.5): #or (err["ON"]["right"] < 0.002 and err["OFF"]["right"] <0.002): # Only for C3, do this for C4
        #if (err["ON"]["left"] < 0.5 and err["OFF"]["left"] <0.5) or (err["ON"]["right"] < 0.5 and err["OFF"]["right"] <0.5):
                #print(corr)
                if list(individual) not in ans["individuals"]:
                        ans["params"].append(params)
                        ans["individuals"].append(sols)
                        #ans["errors"].append(corr)	
                        ans["correlation"].append(corr)	
                        print("solution found !")
                        print(params)			
                        #ans["stats"] = statsVsIters
                        pickle.dump(ans,open(path+"Combinations_"+str(seed1)+".pickle","wb"))

    elif "Bolelli" in params["name"]: # Difficult subject
        #if (err["ON"]["left"] < 0.002 and err["OFF"]["left"] <0.002): #or (err["ON"]["right"] < 0.002 and err["OFF"]["right"] <0.002): # Only for C3, do this for C4
        if (corr["ON"]["left"] >= 0.5 and corr["OFF"]["left"] >=0.5): #or (err["ON"]["right"] < 0.002 and err["OFF"]["right"] <0.002): # Only for C3, do this for C4
        #if (err["ON"]["left"] < 0.5 and err["OFF"]["left"] <0.5) or (err["ON"]["right"] < 0.5 and err["OFF"]["right"] <0.5):
                #print(corr)
                if list(individual) not in ans["individuals"]:
                        ans["params"].append(params)
                        ans["individuals"].append(sols)
                        #ans["errors"].append(corr)	
                        ans["correlation"].append(corr)	
                        print("solution found !")
                        print(params)			
                        #ans["stats"] = statsVsIters
                        pickle.dump(ans,open(path+"Combinations_"+str(seed1)+".pickle","wb"))


    else:
        #if (err["ON"]["left"] < 0.005 and err["OFF"]["left"] <0.001): # or (err["ON"]["right"] < 0.005 and err["OFF"]["right"] <0.001):
        if (corr["ON"]["left"] >= 0.5 and corr["OFF"]["left"] >=0.5): # or (err["ON"]["right"] < 0.005 and err["OFF"]["right"] <0.001):
        #if (err["ON"]["left"] < 0.5 and err["OFF"]["left"] <0.5) or (err["ON"]["right"] < 0.5 and err["OFF"]["right"] <0.5):

                if list(individual) not in ans["individuals"]:
                        ans["params"].append(params)
                        ans["individuals"].append(sols)
                        #ans["errors"].append(corr)	
                        ans["correlation"].append(corr)	
                        print("solution found !")
                        print(params)			
                        #ans["stats"] = statsVsIters
                        pickle.dump(ans,open(path+"Combinations_"+str(seed1)+".pickle","wb"))
    #return Errors,
    #err_num = err["ON"]["left"]+ err["OFF"]["left"]+err["ON"]["right"]+ err["OFF"]["right"]
    corr_num = corr["ON"]["left"]+ corr["OFF"]["left"]#+err["ON"]["right"]+ err["OFF"]["right"]
    #return err_num,
    return corr_num,


def paramSearch(subject,seed,anti=0):

    if os.path.isdir(data_target_dir+str(seed)+"/"+subject) == False:
            os.mkdir(data_target_dir+str(seed)+"/"+subject)

    global simtime 
    simtime = 5000
    global stn_fft
    stn_fft = dict()
    global subject1 
    subject1 = subject
    #global seed
    #seed = seed

    #ffts_dict = pickle.load(open(data_target_dir+"ffts_dict.pickle","rb"))
    #ctx_elec1,stn_elec1 = anal.find_highest_correlated_electrode_pair(ffts_dict[subject],"C3") 

    spec_entropy = pickle.load(open(data_target_dir+"spectral_entropies.pickle","rb"))
    ctx_elec1,stn_elec1 = spec_entropy[subject1]["CTX_fit"], spec_entropy[subject1]["STN_fit"]

    ch_ctx = ctx_elec1
    print(ctx_elec1,stn_elec1)
    for state in ["ON","OFF"]:
            #orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in STN_electrodes if ch in list(subject_data[subject][state].keys())]
            #orig_ts = [   (ch, subject_data[subject][state][ch]) for ch in list(subject_data[subject][state].keys()) if ch == stn_elec1 ]
            orig_ts = []
            for ch in list(subject_data[subject][state].keys()):
                if ch in stn_elec1:
                        orig_ts.append((ch, subject_data[subject][state][ch]))
            #pdb.set_trace()
            stn_fft[state] = dict()
            temp_l = []
            temp_r = []
            temp_chosen = []
            for i,(ch,ts) in enumerate(orig_ts):
                    ts1 = ts['ts'].T[0][500:int(simtime)]
                    #ts1_filt = anal.remove_band_signal(ts1,50,2048)
                    ts1_filt = anal.calc_filtered_signal(ts1,[10.,35.],2048) # Beta band
                    
                    freq,fft = anal.calc_fft(ts1_filt)
                    #if "L" in ch:
                    #    temp_l.append(fft)
                    #elif "R" in ch:
                    #    temp_r.append(fft)
                    
                    #if ch == stn_elec1:
                    temp_chosen.append(fft)


            #stn_fft[state]["left"] = np.mean(temp_l,axis=0)[:int(len(freq)/2)]
            #stn_fft[state]["right"] = np.mean(temp_r,axis=0)[:int(len(freq)/2)]
            ind_freq = np.logical_and(freq>=0,freq<=60)
            stn_fft[state]["freq"] = freq[ind_freq]
            stn_fft[state]["ctx_electrode"] = ctx_elec1
            stn_fft[state]["stn_electrode"] = stn_elec1
            stn_fft[state]["chosen"] = temp_chosen[0][ind_freq]

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
    ans["individuals"] = []
    #ans["errors"] = []
    ans["correlation"] = []

    def findUniq(sols,errs):
            Allstrs = [ np.array2string(np.array(seq)) for seq in sols]
            s,uniqueIds = np.unique(Allstrs, return_index=True)
            uniqSols = np.array(sols)[uniqueIds]
            uniqErrs = np.array(errs)[uniqueIds]

            return uniqSols, uniqErrs

    terms = ["bkg_gpe","bkg_stn","scale_gpe_inh","scale_synaptic","scale_conn","scale_delays"]

    def get_ind():
            return np.random.randint(0,10,1)[0] 


    def initIndividual(icls, content):
            return icls(content)

    def initPopulation(pcls, ind_init, filename):
            contents = pickle.load(open(filename,"rb"))
            return pcls( ind_init(c) for c in contents["params"])
        

    #creator.create("FitnessMin",base.Fitness, weights=(-1.0,))
    creator.create("FitnessMax",base.Fitness, weights=(1.0,))
    #creator.create("Individual",list, fitness=creator.FitnessMin)
    creator.create("Individual",list, fitness=creator.FitnessMax)
    #import pdb
    #pdb.set_trace()

    toolbox = base.Toolbox()
    #toolbox.register("attribute", get_individual )
    toolbox.register("attr_ind",get_ind)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                             (toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind, toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind,toolbox.attr_ind), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_guess",initPopulation,list, toolbox.individual_guess,path+"Combinations_"+str(seed1)+".pickle")

    toolbox.register("mate", tools.cxUniform,indpb=0.2)
    #toolbox.register("mutate", tools.mutGaussian)
    #toolbox.register("mutate", tools.mutUniformInt,low=0,up=9,indpb=0.8)
    toolbox.register("mutate", tools.mutShuffleIndexes,indpb=0.3)
    #toolbox.register("mutate", tools.mutPolynomialBounded,eta=0.5)
    toolbox.register("select", tools.selTournament, tournsize=5)
    #toolbox.register("select", tools.selBest)
    toolbox.register("evaluate", fitness_function)

    #toolbox.decorate("mutate",checkBounds(params["negMax"]*2.5,0,params["posMax"]*1.5))	
    #toolbox.decorate("mate",checkBoundsCrossover(params["negMax"]*2.5,0,params["posMax"]*1.5))	
    pop = toolbox.population(n=20)
    #pop = toolbox.population_guess()
    #pop = toolbox.population(n=10)
    #print pop
    CXPB, MUTPB, NGEN = 0.95, 0.8, 150
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
        #the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        print(offspring)  
        #crossover and mutation on the offspring
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
        if len(ans["params"]) >=10:
            break

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






