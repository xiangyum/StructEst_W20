import pandas as pd
import numpy as np
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

df = pd.read_csv("data/hh_inc_synth.txt", header=None)

def data_moments(df):
    q1_bins = [0,
    4999, 9999, 
    14999, 19999, 
    24999, 29999, 
    34999, 39999, 
    44999, 49999, 
    54999, 59999,
    64999, 69999, 
    74999, 79999,
    84999, 89999, 
    94999, 99999, 
    104999, 109999, 
    114999, 119999, 
    124999, 129999, 
    134999, 139999, 
    144999, 149999, 
    154999, 159999, 
    164999, 169999, 
    174999, 179999, 
    184999, 189999, 
    194999, 199999, 
    249999, 350000]
    q1_bins = [num/1000 for num in q1_bins]
    hist, bin_edges = np.histogram(df, q1_bins)
    '''
    for index, row in data_moments.iterrows():
        if index == len(data_moments-1):
            break
        else:
            lv = bin_edges[index]
            rv = bin_edges[index+1]
            data_moments.loc[index, "lower_bound"] = lv
            data_moments.loc[index, "upper_bound"] = rv

            data_moments.loc[index, "mid_pt"] = (rv + lv)/2
    '''
    return hist, bin_edges
        


def create_midpoints(df):
    q1_bins = [0,
    4999, 9999, 
    14999, 19999, 
    24999, 29999, 
    34999, 39999, 
    44999, 49999, 
    54999, 59999,
    64999, 69999, 
    74999, 79999,
    84999, 89999, 
    94999, 99999, 
    104999, 109999, 
    114999, 119999, 
    124999, 129999, 
    134999, 139999, 
    144999, 149999, 
    154999, 159999, 
    164999, 169999, 
    174999, 179999, 
    184999, 189999, 
    194999, 199999, 
    249999, 350000]
    q1_bins = [num/1000 for num in q1_bins]
    hist, bin_edges = np.histogram(df, q1_bins)
    mmdf = pd.DataFrame({"left_bound": bin_edges[:-1],
                    "right_bound":bin_edges[1:]})
    mmdf["mid"] = (mmdf.left_bound + mmdf.right_bound)/2
    
    return mmdf["mid"]

def lognormal_pdf(x, mu, sigma):
    pdf = 1/x/sigma/np.sqrt(2*np.pi)*np.exp(-1*(np.log(x)-mu)**2/2/sigma**2)
    return pdf

def model_moments(mu, sigma, bin_edges):
    total_n = 121085
    mmdf = pd.DataFrame({"left_bound": bin_edges[:-1],
                    "right_bound":bin_edges[1:]})
    fx = lambda x: lognormal_pdf(x, mu, sigma)
    for index, row in mmdf.iterrows():
        (model_cdf, trash) = intgr.quad(fx, row["left_bound"], row["right_bound"], limit = 200)
        mmdf.loc[index, "cdf"] =  model_cdf
    mmdf["n"] = mmdf.cdf*total_n
    return mmdf.n

def err_vec(xvals, mu, sigma, simple):
    data_vec, bin_edges = data_moments(xvals)
    model_vec = model_moments(mu, sigma, bin_edges)
    if simple:
        err_vec = model_vec - data_vec
    else:
        err_vec = (model_vec - data_vec) / model_vec
    
    return err_vec

def criterion(params, *args):
    mu, sigma = params
    xvals = args
    hist, trash = data_moments(xvals)
    W = np.diag(hist)
    err = err_vec(xvals, mu, sigma, simple=True)
    crit_val = err.T @ W @ err
    return crit_val

mu_init = np.mean(df.hhi)
sigma_init = np.sqrt(np.var(df.hhi))
params_init = np.array([mu_init, sig_init])
gmm_args = (df.hhi)
results = opt.minimize(criterion, 
    params_init, 
    args=(gmm_args), 
    tol=1e-14,
    method='L-BFGS-B', 
    bounds=((1e-10, None), (None, None)))
mu_GMM1, sig_GMM1 = results.x
print('mu_GMM1=', mu_GMM1, ' sig_GMM1=', sig_GMM1)
results
