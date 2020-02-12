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
from scipy.stats import gamma

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

    return hist/len(df), bin_edges
        


def create_midpoints(df):
    q1_bins = [0,
    4999, 9999, 
    14999, 19999, 
    24999, 29999, s
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
    mmdf = pd.DataFrame({"left_bound": bin_edges[:-1],
                    "right_bound":bin_edges[1:]})
    fx = lambda x: lognormal_pdf(x, mu, sigma)
    for index, row in mmdf.iterrows():
        (model_cdf, trash) = intgr.quad(fx, row["left_bound"], row["right_bound"], limit = 200)
        mmdf.loc[index, "cdf"] =  model_cdf
    return mmdf.cdf

def err_vec(xvals, mu, sigma, simple):
    data_vec, bin_edges = data_moments(xvals)
    model_vec = model_moments(mu, sigma, bin_edges).to_numpy()
    if simple:
        err_vec = model_vec - data_vec
    else:
        err_vec = (model_vec - data_vec) / model_vec
    
    return err_vec


def criterion(params, args):
    mu, sigma = params
    xvals = args[0]
    simp = args[1]
    hist, trash = data_moments(xvals)
    W = np.diag(hist)
    err = err_vec(xvals, mu, sigma, simple=simp)
    crit_val = err.T @ W @ err
    return crit_val

mu_init = np.log(np.mean(df.hhi))
sigma_init = 1
params_init = np.array([mu_init, sigma_init])
gmm_args = [df.hhi, False]
results_v1 = opt.minimize(criterion, 
    params_init, 
    args=(gmm_args), 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((1e-10, None), (None, None)))
mu_GMM1_v1, sig_GMM1_v1 = results_v1.x
print("mu_init is {} and sigma_init is {}".format(mu_init, sigma_init))
print('mu_GMM1=', mu_GMM1_v1, ' sig_GMM1=', sig_GMM1_v1)
print("\nValue of Minimized Criterion: {}".format(criterion((mu_GMM1_v1, sig_GMM1_v1), gmm_args)))
results_v1


def GA_pdf(x, alpha, beta):
    # my hw2: 
    # pdf =  1/(beta**(alpha) * math.gamma(alpha))*x**(alpha-1)*np.exp(-x/beta)
    # wiki:
    pdf =  beta**alpha*x**(alpha-1)*np.exp(-beta*x)/math.gamma(alpha)
    return pdf

def GAmodel_moments(alpha, beta, bin_edges):
    mmdf = pd.DataFrame({"left_bound": bin_edges[:-1],
                    "right_bound":bin_edges[1:]})
    fx = lambda x: GA_pdf(x, alpha, beta)
    for index, row in mmdf.iterrows():
        (model_cdf, trash) = intgr.quad(fx, row["left_bound"], row["right_bound"], limit = 200)
        mmdf.loc[index, "cdf"] =  model_cdf
    return mmdf.cdf


def GAerr_vec(xvals, alpha, beta, simple):
    data_vec, bin_edges = data_moments(xvals)
    model_vec = GAmodel_moments(alpha, beta, bin_edges).to_numpy()
    if simple:
        err_vec = model_vec - data_vec
    else:
        err_vec = (model_vec - data_vec) / data_vec
    
    return err_vec

def GAcriterion(params, args):
    alpha, beta = params
    xvals = args[0]
    simp = args[1]
    hist, trash = data_moments(xvals)
    W = args[2]
    err = GAerr_vec(xvals, alpha, beta, simple=simp)
    crit_val = err.T @ W @ err
    return crit_val

hist, bin_edges = data_moments(df.hhi)
alpha_init = 3
beta_init = 1/20000
params_init = np.array([alpha_init, beta_init])
gmm_args = [df.hhi, False, np.diag(hist)]
ga_results_v1 = opt.minimize(GAcriterion, 
    params_init, 
    args=gmm_args, 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((1e-7, None), (1e-7, None)))
GA_alpha_GMM1_v1, GA_beta_GMM1_v1 = ga_results_v1.x
print("\nalpha_init is {} and beta_init is {}".format(alpha_init, beta_init))
print("\nValue of Minimized Criterion (Initial) : {}".format(criterion((alpha_init, beta_init), gmm_args)))
print('\nalpha_GMM1=', GA_alpha_GMM1_v1, 'beta_GMM1=', GA_beta_GMM1_v1)
print("\nValue of Minimized Criterion (Final): {}".format(criterion((GA_alpha_GMM1_v1, GA_beta_GMM1_v1), gmm_args)))
ga_results_v1



def get_errmatrix(xvals, alpha, beta, simple= False):
    data_vec, bin_edges = data_moments(xvals)
    R = 42
    N = len(xvals)
    Err_mat = np.zeros((R, N))
    model_vec = GAmodel_moments(alpha, beta, bin_edges)
    mmdf = pd.DataFrame({"left_bound": bin_edges[:-1],
                    "right_bound":bin_edges[1:]})
    if simple:
        for index, row in mmdf.iterrows():
            lb = row["left_bound"]
            rb = row["right_bound"]
            pts_in_grp = (lb <= xvals) & (xvals < rb)
            Err_mat[index, :] = pts_in_grp- model_vec[index]
    else:
        for index, row in mmdf.iterrows():
            lb = row["left_bound"]
            rb = row["right_bound"]
            pts_in_grp = (lb <= xvals) & (xvals < rb)
            Err_mat[index, :] = (pts_in_grp- model_vec[index])/model_vec[index]
    return Err_mat

emt = get_errmatrix(df.hhi, GA_alpha_GMM1_v1, GA_beta_GMM1_v1)
omega1 = 1/len(df) * emt @ emt.T
new_w = lin.pinv(omega1)
hist, bin_edges = data_moments(df.hhi)
alpha_init = 3
beta_init = 1/20000
params_init = np.array([alpha_init, beta_init])
gmm_args = [df.hhi, False, new_w]
ga_results_v1 = opt.minimize(GAcriterion, 
    params_init, 
    args=gmm_args, 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((1e-7, None), (1e-7, None)))
GA_alpha_GMM1_v1, GA_beta_GMM1_v1 = ga_results_v1.x
print("\nalpha_init is {} and beta_init is {}".format(alpha_init, beta_init))
print("\nValue of Minimized Criterion (Initial) : {}".format(criterion((alpha_init, beta_init), gmm_args)))
print('\nalpha_GMM1=', GA_alpha_GMM1_v1, 'beta_GMM1=', GA_beta_GMM1_v1)
print("\nValue of Minimized Criterion (Final): {}".format(criterion((GA_alpha_GMM1_v1, GA_beta_GMM1_v1), gmm_args)))
ga_results_v1
