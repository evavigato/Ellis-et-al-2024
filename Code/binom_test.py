# %%
import pandas as pd
from scipy.stats import binomtest
import numpy as np

# %%

def preprocess(raw_W):
    W = raw_W.copy()
    to_remove = ['dl6', 'vm6']
    for col in W.columns:
        if col == 'other' or col == 'other ':
            to_remove.append(col)
        if 'Unnamed' in col:
            to_remove.append(col)
    W = W.drop(columns=to_remove)
    W = W.rename(columns={"vc3": "vc3l", 'vm7':'vm7d', '1':'vm7v'})
    W = W.sort_index(axis='columns')
    W = W.replace(np.nan, int(0))
    W = W > 0
    W = W.astype(int)
    return W

# %%
# Different conn mats to test
W_Ellis_Mel_F = pd.read_csv('Ellis_raw_D_Mel_Female.csv')
W_Ellis_Mel_F_ab = preprocess(W_Ellis_Mel_F[W_Ellis_Mel_F['Unnamed: 2'] == 'alpha/beta'])
W_Ellis_Mel_F_abp = preprocess(W_Ellis_Mel_F[W_Ellis_Mel_F['Unnamed: 2'] == 'alpha\'/beta\''])
W_Ellis_Mel_F_gamma = preprocess(W_Ellis_Mel_F[W_Ellis_Mel_F['Unnamed: 2'] == 'gamma'])
W_Ellis_Mel_F = preprocess(W_Ellis_Mel_F)


W_Ellis_Mel_M = pd.read_csv('Ellis_raw_D_Mel_Male.csv')
W_Ellis_Mel_M_ab = preprocess(W_Ellis_Mel_M[W_Ellis_Mel_M['Unnamed: 2'] == 'alpha/beta'])
W_Ellis_Mel_M_abp = preprocess(W_Ellis_Mel_M[W_Ellis_Mel_M['Unnamed: 2'] == 'alpha\'/beta\''])
W_Ellis_Mel_M_gamma = preprocess(W_Ellis_Mel_M[W_Ellis_Mel_M['Unnamed: 2'] == 'gamma'])
W_Ellis_Mel_M = preprocess(W_Ellis_Mel_M)

W_Ellis_Sech = pd.read_csv('Ellis_raw_D_Sech.csv')
W_Ellis_Sech_ab = preprocess(W_Ellis_Sech[W_Ellis_Sech['Unnamed: 2'] == 'alpha/beta'])
W_Ellis_Sech_abp = preprocess(W_Ellis_Sech[W_Ellis_Sech['Unnamed: 2'] == 'alpha\'/beta\''])
W_Ellis_Sech_gamma = preprocess(W_Ellis_Sech[W_Ellis_Sech['Unnamed: 2'] == 'gamma'])
W_Ellis_Sech = preprocess(W_Ellis_Sech)

W_Ellis_Sim = pd.read_csv('Ellis_raw_D_Sim.csv')
W_Ellis_Sim_ab = preprocess(W_Ellis_Sim[W_Ellis_Sim['Unnamed: 2'] == 'alpha beta'])
W_Ellis_Sim_abp = preprocess(W_Ellis_Sim[W_Ellis_Sim['Unnamed: 2'] == 'alpha\' beta\''])
W_Ellis_Sim_gamma = preprocess(W_Ellis_Sim[W_Ellis_Sim['Unnamed: 2'] == 'gamma'])
W_Ellis_Sim = preprocess(W_Ellis_Sim)


# %%
def ACP(W):
    acp = np.mean(W>0,0)
    acp = acp / np.sum(acp)

    return acp

# %%
def p_values(W, title):
    p = ACP(W).mean()
    count_sums = W.sum()
    total = count_sums.sum()

    pvals = []
    for i_glom, sum in enumerate(count_sums):
        p_value = binomtest(sum, total, p)
        pvals.append(p_value.pvalue)

    results = pd.DataFrame({'Glomerulus': ACP(W).index, 'Connection probability': list(ACP(W)), 'P-values': pvals})
    results = results.sort_values('Connection probability')
    results.to_csv('pvals_'+title+'.csv')

    return results

# %%
results = p_values(W_Ellis_Sim_gamma, 'Sim_Gamma')


