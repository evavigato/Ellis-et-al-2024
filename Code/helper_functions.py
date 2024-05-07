# Imports
import numpy as np
import pandas as pd   
import pickle
from matplotlib import pyplot as plt
import scipy.stats as st
from scipy import linalg
import math
import warnings 
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
import random
from scipy.spatial import *
import seaborn as sns
from scipy.spatial import distance
from sklearn.preprocessing import binarize



def get_Caronlike(file):
	Caron = pd.read_csv(file)
	headers = list(Caron.columns.values)

	W_Caron = Caron.iloc[:, 5:]
	headers = list(W_Caron.columns.values)
	W_Caron = W_Caron.fillna(0.0)
	W_Caron = W_Caron.sort_index(axis=1)
	return W_Caron, Caron



# Shuffle matrix returning shuffled matrix mat_W with indegree of mat_X, keeping connection probs consistent
# shufmat
def fixed(mat_W, mat_X):
    M = mat_W.shape[0]
    N = mat_W.shape[1]

    indeg = np.sum(mat_X>0,1)
    
    cprobs = np.mean(mat_W>0,0)
    cprobs = cprobs / np.sum(cprobs)

    Wshuf = np.zeros([M,N])

    for mi in range(M):
        num_inputs = np.random.choice(indeg)
        inds = np.random.choice(N,num_inputs,p=cprobs, replace=False)
        Wshuf[mi,inds] = 1

    return Wshuf

# Shuffle matrix returning shuffled matrix mat_W with indegree of mat_X
# shufmat_indegree_only
def shuffle(mat_W, mat_X):
    M = mat_W.shape[0]
    N = mat_W.shape[1]

    indeg = np.sum(mat_X>0,1)

    Wshuf = np.zeros([M,N])

    for mi in range(M):
        num_inputs = np.random.choice(indeg)
        inds = np.random.choice(N,num_inputs,replace=False)
        Wshuf[mi,inds] = 1

    return Wshuf


# 	return acps

def plot_ACP_updated(list_weights, list_labels, title, output=True):
	acps = [] # list of all acps
	acp_data=[] # data to csv

	for i in range(len(list_weights)):
		W = list_weights[i]
		acp = np.mean(W>0,0)
		acp = acp / np.sum(acp)
		acps.append(acp)

	acpsum = sum(acps).values

	indsort = np.argsort(acpsum)[::-1]
        
	if output==True:

		# Create figure
		acp_fig = plt.figure(figsize=(15,10))
		ax1 = acp_fig.add_subplot(111)

		for j in range(len(list_weights)):
			acp = acps[j]
			ax1.plot(acp.values[indsort], marker='.', label=list_labels[j])
			acp_data.append(acp.values[indsort])


		plt.xticks(np.arange(len(acp.values)),acp.index.values[indsort],rotation=90, fontsize=15)
		plt.ylabel('average connection probability')
		plt.title(title)
		plt.legend()

		plt.show()

		# Generate CSV
		output = pd.DataFrame(data=np.array(acp_data).T, columns=list_labels)
		output.index=acp.index.values[indsort]
		output.to_csv(title+'.csv')


	return acps, output



def confidence_interval(a):
    sorted = np.sort(a)
    boundary = int(np.around(0.026*len(a)))
    lower = sorted[boundary]
    higher = sorted[-boundary]
    return lower, higher


def jsanalysis(weight_mats, labels, title, to_plot=True):
    acps, _ = plot_ACP_updated(weight_mats, labels, 'all_connection_probabilities', output=False)
    
    dists_sj = np.ndarray(shape=(len(weight_mats), len(weight_mats)))
    for i, acpA in enumerate(acps):
        for j, acpB in enumerate(acps):
            dists_sj[i, j] = distance.jensenshannon(acpA,acpB)
    
    if to_plot == True:
        plt.figure(figsize=(35, 10))
        # sns.set(font_scale=2)
        sns.heatmap(dists_sj, annot=True, xticklabels=labels, yticklabels=labels, fmt='.2g')

    # save data to csv
    df = pd.DataFrame(dists_sj)
    df.columns = labels
    df.index = labels
    df.to_csv('js_distances.csv')

    return df

def fixed_pn_fixed_kc(W):
    """ Fix pn out-degree and kc in-degree

    Args:
        W (_type_): connectivity matrix (PN x KC)

    Returns:
        _type_: shuffled binary matrix 
    """    
    # Binarize connectivity
    mat_W = binarize(np.array(W))

    n_PNs = mat_W.shape[0]
    n_KCs = mat_W.shape[1]

    # In-degree to the KCs
    indeg = np.sum(mat_W>0,0)

    # Connection probability of PNs
    cprobs = np.mean(mat_W>0,1) # conn probabilities for each PN
    cprobs = cprobs / np.sum(cprobs) # so that all connection probabilities add up to 1

    Wshuf = np.zeros([n_PNs,n_KCs])

    for kc in range(n_KCs):
        num_inputs = np.random.choice(indeg) # use observed in-degrees to kcs
        inds = np.random.choice(n_PNs,num_inputs,p=cprobs, replace=False) # pick incident PNs without replacement, using observed connection probabilities 
        Wshuf[inds, kc] = 1

    return Wshuf



