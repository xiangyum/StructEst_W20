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
import math

mdf = pd.read_csv("data/MacroSeries.txt", header=None)
mdf["c"] = mdf[0]
mdf["k"] = mdf[1]
mdf["w"] = mdf[2]
mdf["r"] = mdf[3]
mdf = mdf.loc[:,["c", "k", "w", "r"]]

def p2_model_moments(df, alpha, rho, mu):
    # construct Zs
    beta = 0.99
    intial_r = df.r[0]
    initial_k = df.k[0]
    initial_z = np.log(intial_r/alpha/initial_k**alpha)
    df.loc[0, "z"] = initial_z
    for i in range(1, len(df)):
        prev_z = df.loc[i-1, "z"]
        new_z = rho*prev_z + (1-rho)*mu
        df.loc[i, "z"] = new_z

    for index, row in df.iterrows():
        if index < len(df)-1:
            # m1
            row2 = df.loc[index+1,:]
            m1 = row2.z - rho*row.z - (1-rho)*mu
            df.loc[index, "m1"] = m1

            # m2
            m2 = (row2.z- rho*row.z - (1-rho)*mu)*row.z
            df.loc[index, "m2"] = m2

            # m3
            m3 =  beta*alpha*np.exp(row2.z)*row2.k**(alpha-1)*row.c/row2.c-1
            df.loc[index, "m3"] = m3

            # m4
            m4 = (beta*alpha*np.exp(row2.z)*row2.k**(alpha-1)*row.c/row2.c-1)*row.w
            df.loc[index, "m4"] = m4

    mean_m1 = np.mean(df.m1)
    mean_m2 = np.mean(df.m2)
    mean_m3 = np.mean(df.m3)
    mean_m4 = np.mean(df.m4)


    return np.array([mean_m1, mean_m2 ,mean_m3 ,mean_m4])

def p2_err_vec(xvals, alpha, rho, mu):
    data_vec = np.array([0, 0, 0 ,0])
    model_vec = p2_model_moments(xvals, alpha, rho, mu)
    err_vec = model_vec - data_vec
    return err_vec

def p2_criterion(params, args):     
    alpha, rho, mu = params
    xvals = args[0]
    W = np.eye(4)
    err = p2_err_vec(xvals, alpha, rho, mu)
    crit_val = err.T @ W @ err
    return crit_val

alpha_init = 0.6
rho_init = 0.6
mu_init = 2
params_init = np.array([alpha_init, rho_init, mu_init])
gmm_args = [df]
results_v1 = opt.minimize(p2_criterion, 
    params_init, 
    args=(gmm_args), 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((1e-10, 1-1e-10), (-1+1e-10, 1-1e-10), (1e-10, None)))

alpha_GMM, rho_GMM, mu_GMM = results_v1.x
print("\n alpha_init = {}, rho_init = {}, and mu_init = {}".format(alpha_init, rho_init, mu_init))
print("\n alpha_GMM = {}, rho_GMM = {}, and mu_GMM = {}".format(alpha_GMM, rho_GMM, mu_GMM))
print("\nValue of Minimized Criterion: {}".format(p2_criterion((alpha_GMM, rho_GMM, mu_GMM), gmm_args)))
results_v1






