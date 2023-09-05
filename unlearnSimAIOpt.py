#This file contains the simulation code for the paper Human-AI Delegation in Organizational Routines
#Code for generating plots is not inlcuded
#It will be publicly released, please do not redistribute prior to acceptance of the manuscript


import os
import numpy as np
import pickle

#Simulate an iteration
# Inputs:
# pAI ... probability to use AI for each action
# V_hu... Variation of human per action
# V_ai... Variation of AI per action
# Outputs:
#     ntries = np.ones(L) #tries for each action, by default do just one try
#     ntriesAI =np.zeros(L) #how often was AI used for an action
#     nIss = np.zeros(L) #issues overall per action
#     nIssAI = np.zeros(L)  #issues when AI was used per action

def generate_S(pAI, V_hu,  V_ai):

    #Initialize variables
    L= len(pAI) #number of actions
    ntries = np.ones(L) #tries for each action, by default do just one try
    ntriesAI =np.zeros(L) #how often was AI used for an action
    nIss = np.zeros(L) #issues overall per action
    nIssAI = np.zeros(L)  #issues when AI was used per action

    for next_step in range(L): #walk through the process action by action

        isAI = np.random.random() < pAI[next_step] #Randomly choose if delegate to AI or not
        if isAI: ntriesAI[next_step]+=1 #inc count if delegate to AI

        # Complete action, i.e. (re)try until there is no issue
        while True:
            #Determine if there is an issue
            issueRa = np.random.random() #Choose a random number determining if there is an issue
            issue= ((not isAI and issueRa < V_hu[next_step]) or (isAI and issueRa < V_ai[next_step])) #there is an issue, if don't use AI and random val is below issue probability of humans; if use AI there is an issue if random number is below probability of issue of AI

            if issue: # There is an issue, choose another action
                ntries[next_step] += 1 #One more try is needed since there is an issue
                nIss[next_step]+=1 #increment number of issues
                if isAI: nIssAI[next_step]+=1 #increment AI issues if AI was used
                isAI = np.random.random() < 0.5 #Retry by choosing randomly either AI or not (as in Pentland)
                if isAI: ntriesAI[next_step] += 1 #increment count of AI tries
            else:
                break #no issue stop!
    return ntries, ntriesAI, nIss,nIssAI



#get probability to use AI for each action
#Input:
#SAI...organizational memory stating how often was AI used for an action within the remembered past iterations
#nIssAI...organizational memory stating how often AI failed for an action within the remembered past iterations
#Output:
#pAI...Probability for AI

def get_pAI(SAI,nIssAI):
    NAI= np.sum(SAI,axis=0) #how often did AI fail for each action across all rememberd iterations
    sucAI = NAI-np.sum(nIssAI,axis=0) #how often was AI successful for each action, i.e., either 0 (if it was tried and failed) or 1 (since an action with an iteration is only retired until it suceeds once)
    pAI=sucAI/len(SAI) #compute probability to use AI per action as successes/possible tries
    return pAI


#Simulate one run, i.e. all iterations, for a specific parameter setting
#cfg is a dictionary containing all parameter values
def trainOne(cfg,a,b):

    #set parameters
    N = cfg["N"] #Iterations
    L  = cfg["L"]  # number of possible actions
    V_H = np.full(L, cfg["V"])  # probability of issue  # change of V per Iter
    M = cfg["R"]  # number of past performances


    #initialize variables
    V_ai = np.full(L, cfg["VAI"])  # set initial variation for each action
    S = np.full((N + M, L),1) #Org. Memory: number of executions of an action (sum of AI and non-AI) for all iterations and including the initial memory, where we assume that each action was tried just ones (e.g. there was no rework);
    SAI = np.full((N + M, L),cfg["iHAI"]) #Org. memory: number of executions using AI of an action including initial memory; initially, we assume that for each action a fraction PR_AI is done by AI and that there are no failures, this leads also to an initial probability of PR_AI for each action
    issAI = np.full((N + M, L),0)  #Org. memory number of executions using AI of an action resulting in an issue, where we initialize it to given initial probability to use AI
    tissAI = np.zeros((N,L)).astype(np.uint32) #Issues of AI per action for each iteration
    tiss = np.zeros((N,L)).astype(np.uint32) #All Issues per action for each iteration
    tpAI = np.zeros((N,L)).astype(np.float32) #Probabilities of AI across all iterations for each action
    tdurs = np.zeros((N,L)).astype(np.float32) #Durations ...
    tcompEnt=np.zeros((N,L)).astype(np.float32) #Entropy...
    pAIs = np.zeros(L).astype(np.float32)  # probability to use AI for each action, initialized based on initialy memory S,SAI and issAI

    #Define learning curve
    accGain = cfg["accGain"]  # learning parameters a,b of learning curve
    dV_ai_per_sample = lambda acc: 0 if accGain[0] == 0 else accGain[0] * ((1 - acc) ** accGain[1])  # Learning curve, i.e., how much variation changes with one data sample


    # generate iterations
    for t in range(N):

        # Update/set  probabilities
        if M > 0: pAIs = get_pAI(SAI[t: (t + M)], issAI[t: (t + M)])  # get current probabilities

        # Sanity check that probabilities are valid - never failed, but it is still good practice to keep tests in code
        if np.max(pAIs)>1 or np.min(pAIs)<0:
            print("Fatal probability larger 1 or smaller 0!",t,np.max(pAIs),np.min(pAIs),cfg["pr"])
            os.exit(-1)

        # generate an iteration and store it in organizational memory
        newS,newSAI,newIss,newIssAI= generate_S(pAIs, V_H, V_ai)

        #Store new iteration data
        tiss[t] = newIss #store other issues
        tissAI[t] = newIssAI #store AI issues
        tpAI[t] = pAIs  # store probabilities

        #Compute duration
        tdurs[t] =  ((newS - newSAI) + cfg["aiDur"] *  newSAI)  # Duration to complete action (computed for each action in parallel), this includes time for rework
        #print("\nDur",np.mean(tdurs[t]),newS[:10],newSAI[:10])
        #print(" ", V_ai[:5],V_H[:5])

        #compute entropy
        entropy = -( (1-pAIs)*np.log2(1-pAIs+1e-10) + pAIs*np.log2(pAIs+1e-10) ) #Shannon's entropy, use small constant to avoide np.log(0) which yields infty (np.inf) and can be problematic in further computations
        tcompEnt[t]=entropy

        #Update variation of AI
        dV_ai = newSAI * dV_ai_per_sample(V_ai) #compute total change of variation of AI. It is given by the change for one sample (based on learning curve) multiplied with the number of tries (nAI)
        V_ai = np.clip(V_ai-dV_ai,0,10000) #must clip, since otherwise V_ai might become slightly negative

        # Update memory,i.e., replace oldest memory with current one, i.e., in next iteration this iteration, will be the last considered
        S[t + M] = newS
        SAI[t + M] = newSAI
        issAI[t + M] = newIssAI


    #Compute final result
    mr= lambda x: list(np.round(np.mean(x,axis=1),2)) #aggregate measures accross all actions and round a bit (saves a lot of space / time but has almost no impact on accuracy), saving as numpy array would be even better
    mcompEnt=mr(tcompEnt)
    mdurs = mr(tdurs)
    miss = mr(tiss)
    mpai = mr(tpAI)
    if "onlyPrint" in cfg:
        print("Done")
    else:
        #Store result
        import benchHelp
        benchHelp.getPaths(cfg, longName=False, create=True, setRand=True, pathAdd="")
        cfg["result"] = {"coEnt": mcompEnt,"durs": mdurs,"iss": miss,"pAI":mpai}
        print("Done",cfg["pr"])
        with open(cfg["bFolder"] + "cfg" + str(cfg["bid"]) +".pic", "wb") as f: pickle.dump(cfg, f)