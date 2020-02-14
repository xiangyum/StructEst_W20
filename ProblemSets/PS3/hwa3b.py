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

def p2_data_moments(df, alpha, rho, mu):
    # construct Zs
    random.seed(47)
    beta = 0.99
    for index, row in df.iterrows():
        intial_r = df.r[index]
        initial_k = df.k[index]
        initial_z = np.log(intial_r/alpha/initial_k**(alpha-1))
        df.loc[index, "z"] = initial_z
    
    m1 = 0
    m2 = 0
    m3 = 0
    m4 = 0 
    counter = 1

    for index, row in df.iterrows():
        if index < len(df)-1:
            # m1
            row2 = df.loc[index+1,:]
            m1 += row2.z - rho*row.z - (1-rho)*mu

            # m2
            m2 += (row2.z- rho*row.z - (1-rho)*mu)*row.z
            
            # m3
            m3 +=  beta*alpha*np.exp(row2.z)*row2.k**(alpha-1)*row.c/row2.c-1

            # m4
            m4 += (beta*alpha*np.exp(row2.z)*row2.k**(alpha-1)*row.c/row2.c-1)*row.w
            counter += 1
            
    return np.array([m1/counter, m2/counter ,m3/counter ,m4/counter])

def p2_err_vec(xvals, alpha, rho, mu):
    model_vec = np.array([0, 0, 0 ,0])
    data_vec = p2_data_moments(xvals, alpha, rho, mu)
    err_vec = model_vec - data_vec
    return err_vec

def p2_criterion(params, args):
    alpha, rho, mu = params
    xvals  = args[0]
    W = args[1]
    err = p2_err_vec(xvals, alpha, rho, mu)
    crit_val = err.T @ W @ err
    return crit_val

alpha_init = 0.6
rho_init = 0.6
mu_init = 2
params_init = np.array([alpha_init, rho_init, mu_init])
gmm_args = [df, np.eye(4)]
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



# 2-step

def p2_errmatrix(df, alpha, rho, mu):
    edf = pd.DataFrame()
    beta = 0.99
    for index, row in df.iterrows():
        intial_r = df.r[index]
        initial_k = df.k[index]
        initial_z = np.log(intial_r/alpha/initial_k**(alpha-1))
        df.loc[index, "z"] = initial_z

    for i in range(0, len(df)-1):
        row = df.loc[i,:]
        row2 = df.loc[i+1,:]
        edf.loc["m1", i] = row2.z - rho*row.z - (1-rho)*mu
        edf.loc["m2", i] = (row2.z- rho*row.z - (1-rho)*mu)*row.z
        edf.loc["m3", i] = beta*alpha*np.exp(row2.z)*row2.k**(alpha-1)*row.c/row2.c-1
        edf.loc["m4", i] = (beta*alpha*np.exp(row2.z)*row2.k**(alpha-1)*row.c/row2.c-1)*row.w
    return np.array(edf)

emt = p2_errmatrix(df, alpha_GMM, rho_GMM, mu_GMM)
omega1 = 1/(len(df)-2) * emt @ emt.T
new_w = lin.pinv(omega1)
params_init = np.array([alpha_GMM, rho_GMM, mu_GMM])
gmm_args = [df, new_w]
ga2step_results = opt.minimize(p2_criterion, 
    params_init, 
    args=(gmm_args), 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((1e-10, 1-1e-10), (-1+1e-10, 1-1e-10), (1e-10, None)))
GA2step_alpha_GMM, GA2step_rho_GMM, GA2step_mu_GMM = ga2step_results.x
print("\n alpha_init = {}, rho_init = {}, and mu_init = {}".format(alpha_init, rho_init, mu_init))
print("\n alpha_GMM = {}, rho_GMM = {}, and mu_GMM = {}".format(GA2step_alpha_GMM, GA2step_rho_GMM, GA2step_mu_GMM))
print("\nValue of Minimized Criterion: {}".format(p2_criterion((GA2step_alpha_GMM, GA2step_rho_GMM, GA2step_mu_GMM), gmm_args)))
ga2step_results


def p2_jacob(df, alpha, rho, mu):
    beta = 0.99
    h_alpha = 1e-8 * alpha
    h_rho = 1e-8 * rho
    h_mu = 1e-8 * mu
    evec_alpha = p2_err_vec(df, alpha + h_alpha, rho, mu) - p2_err_vec(df, alpha - h_alpha, rho, mu)
    evec_rho = p2_err_vec(df, alpha , rho + h_rho, mu) - p2_err_vec(df, alpha , rho - h_rho, mu)
    evec_mu = p2_err_vec(df, alpha, rho, mu + h_mu) - p2_err_vec(df, alpha, rho, mu - h_mu)
    jdf = pd.DataFrame({"1": evec_alpha,
                    "2": evec_rho,
                    "3":evec_mu})
    return np.array(jdf)


n = len(df)-1
d_err = p2_jacob(df, GA2step_alpha_GMM, GA2step_rho_GMM, GA2step_mu_GMM)
SigHat2 = (1 / n) * lin.inv(d_err.T @ new_w @ d_err)
print(SigHat2)

print('Std. err. alpha_hat=', np.sqrt(SigHat2[0, 0]))
print('Std. err. rho_hat=', np.sqrt(SigHat2[1, 1]))
print('Std. err. mu_hat=', np.sqrt(SigHat2[2, 2]))

N = pts.shape[0]
d_err2 = Jac_err2(pts, mu_GMM1, sig_GMM1, 0.0, 450.0, False)
print(d_err2)
print(W_hat)
SigHat2 = (1 / N) * lin.inv(d_err2.T @ W_hat @ d_err2)
print(SigHat2)
print('Std. err. mu_hat=', np.sqrt(SigHat2[0, 0]))
print('Std. err. sig_hat=', np.sqrt(SigHat2[1, 1]))








