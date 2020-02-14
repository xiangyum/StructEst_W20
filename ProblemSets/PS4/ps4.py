import numpy as np
import pandas as pd
import random
import scipy.stats as sts
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt

df = pd.read_csv("data/NewMacroSeries.txt", header=None)
df["c"] = df[0]
df["k"] = df[1]
df["w"] = df[2]
df["r"] = df[3]
df["y"] = df[4]
df = df.loc[:,["c", "k", "w", "r", "y"]]

# simulate draws
random.seed(47)
size_n = 100
num_sets = 1000
unif_values = sts.uniform.rvs(0, 2, size = (size_n, num_sets))
drawsDF = pd.DataFrame(unif_values)

# remember: ppf functions map "percentiles" from these uniform draws onto 
#another probability distribution!

# functions
def data_moments(df):
    m1 = np.mean(df.c)
    m2 = np.mean(df.k)
    m3 = np.mean(df.c/df.y)
    m4 = np.var(df.y)
    temp = df.copy()
    for i in range(1, len(df)):
        temp.loc[i,"prev_c"] = temp.loc[i-1,"c"]
    temp = temp.loc[1:, ["c", "prev_c"]].copy()
    m5 = np.corrcoef(temp.c, temp.prev_c)[0,1]
    m6 =  np.corrcoef(df.c, df.r)[0,1]
    return np.array([m1, m2, m3, m4, m5, m6])

def model_moments(df, draws, alpha, rho, mu, sigma):
    beta = 0.99
    k1 = np.mean(df.k)
    mmDF = pd.DataFrame({"set":[i for i in range(0,1000)]})
    for set_num in range(0, 1000):
        tdf = df.copy()
        tdf["eps"] = draws.loc[:,set_num]
        # initialize first z value:
        tdf.loc[0, "z"] = mu
        tdf.loc[0, "sk"] = k1
        for i in range(1, 100-1):
             prev_z = tdf.loc[i-1, "z"]
             prev_k = tdf.loc[i-1, "sk"]
             tdf.loc[i, "z"] = rho*prev_z + (1-rho)*mu + tdf.loc[i, "eps"]
             tdf.loc[i, "sk"] = alpha * beta * np.exp(tdf.loc[i-1, "z"])*prev_z
             tdf.loc[i, "sw"] = (1-alpha)*np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]**alpha
             tdf.loc[i, "sr"] = alpha*np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]**(alpha-1)
             next_k =  alpha * beta * np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]
             tdf.loc[i, "sc"] = tdf.loc[i, "sw"] + tdf.loc[i, "sr"]*tdf.loc[i, "sk"]  - next_k
             tdf.loc[i, "sy"] = np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]**alpha
        
        # model moments
        mmDF.loc[set_num, "m1"] = np.mean(tdf.sc)
        mmDF.loc[set_num, "m2"] = np.mean(tdf.sk)
        mmDF.loc[set_num, "m3"] = np.mean(tdf.sc/tdf.sy)
        mmDF.loc[set_num, "m4"] = np.var(tdf.sy)
        temp = tdf.copy()
        for i in range(1, len(tdf)):
            temp.loc[i,"prev_c"] = temp.loc[i-1,"sc"]
        temp = temp.loc[:, ["sc", "prev_c"]].copy().dropna()
        mmDF.loc[set_num, "m5"] = np.corrcoef(temp.sc, temp.prev_c)[0,1]
        temp = tdf.loc[:, ["sc", "sr"]].copy().dropna()
        mmDF.loc[set_num, "m6"] =  np.corrcoef(temp.sc, temp.sr)[0,1]
    return np.array([np.mean(mmDF.m1), 
        np.mean(mmDF.m2), 
        np.mean(mmDF.m3), 
        np.mean(mmDF.m4), 
        np.mean(mmDF.m5), 
        np.mean(mmDF.m6)])

def err_vec(df, draws, alpha, rho, mu, sigma, simple = False):
    data_vec = data_moments(df)
    model_vec = model_moments(df, draws, alpha, rho, mu, sigma)
    if simple:
        err_vec = model_vec - data_vec
    else:
        err_vec = (model_vec - data_vec)/data_vec

    return err_vec

def criterion(params, args):
    alpha, rho, mu, sigma = params
    df  = args[0]
    W = args[1]
    draws = args[2]
    simple_condition = args[3]
    err = err_vec(df, draws, alpha, rho, mu, sigma, simple_condition)
    crit_val = err.T @ W @ err
    return crit_val


alpha_init = 0.6
rho_init = 0.6
mu_init = 7
sigma_init = 0.5
params_init = np.array([alpha_init, rho_init, mu_init, sigma_init])
gmm_args = [df, np.eye(6), drawsDF, False]
results = opt.minimize(criterion, 
    params_init, 
    args=gmm_args, 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((0.01, 0.99), (-0.99,0.99), (5,14), (0.01, 1.1)))
results


alpha_GMM, rho_GMM, mu_GMM, sigma_GMM = results.x
print("\n alpha_init = {}, rho_init = {}, and mu_init = {}".format(params_init[0], params_init[1],params_init[2],params_init[3]))
print("\n alpha_GMM = {}, rho_GMM = {}, and mu_GMM = {}".format(alpha_GMM, rho_GMM, mu_GMM, sigma_GMM))
print("\nValue of Minimized Criterion: {}".format(criterion((alpha_GMM, rho_GMM, mu_GMM, sigma_GMM), gmm_args)))


def jacobian_err(df, draws, alpha, rho, mu, sigma, simple = False):
    beta = 0.99
    h_alpha = 1e-4 * alpha
    h_rho = 1e-4 * rho
    h_mu = 1e-4 * mu
    h_sigma = 1e-4 * sigma

    alpha_vector = (err_vec(df, draws, alpha + h_alpha, rho, mu, sigma, simple = False) - err_vec(df, draws, alpha - h_alpha, rho, mu, sigma, simple = False))/(2*h_alpha)
    rho_vector = (err_vec(df, draws, alpha, rho + h_rho, mu, sigma, simple = False) - err_vec(df, draws, alpha, rho - h_rho, mu, sigma, simple = False))/(2*h_rho)
    mu_vector = (err_vec(df, draws, alpha , rho, mu+ h_mu, sigma, simple = False) - err_vec(df, draws, alpha , rho, mu-h_mu, sigma, simple = False))/(2*h_mu)
    sigma_vector = (err_vec(df, draws, alpha, rho, mu, sigma+h_sigma, simple = False) - err_vec(df, draws, alpha, rho, mu, sigma-h_sigma, simple = False))/(2*h_sigma)

    jdf = pd.DataFrame({"1": alpha_vector,
        "2": rho_vector,
        "3": mu_vector,
        "4": sigma_vector})
    return np.array(jdf)

d_err = jacobian_err(df, drawsDF, alpha_GMM, rho_GMM, mu_GMM, sigma_GMM, simple = False)

SigHat2 = (1 / 1000) * lin.inv(d_err.T @ np.eye(6) @ d_err)
print(SigHat2)
print('Std. err. alpha_hat=', np.sqrt(SigHat2[0, 0]))
print('Std. err. rho_hat=', np.sqrt(SigHat2[1, 1]))
print('Std. err. mu_hat=', np.sqrt(SigHat2[2, 2]))
print('Std. err. sigma_hat=', np.sqrt(SigHat2[3, 3]))



def err_matrix(df, draws, alpha, rho, mu, sigma):
    # produce a R x N matrix, where R = no. of moments & N = no. of points in the data
    mmDF = pd.DataFrame()
    beta = 0.99
    data_vec = data_moments(df)
    dm1, dm2, dm3, dm4, dm5, dm6 = data_vec[0], data_vec[1], data_vec[2], data_vec[3], data_vec[4], data_vec[5]
    for set_num in range(0, 1000):
        tdf = df.copy()
        tdf["eps"] = draws.loc[:,set_num]
        # initialize first z value:
        tdf.loc[0, "z"] = mu
        tdf.loc[0, "sk"] = k1
        for i in range(1, 100-1):
             prev_z = tdf.loc[i-1, "z"]
             prev_k = tdf.loc[i-1, "sk"]
             tdf.loc[i, "z"] = rho*prev_z + (1-rho)*mu + tdf.loc[i, "eps"]
             tdf.loc[i, "sk"] = alpha * beta * np.exp(tdf.loc[i-1, "z"])*prev_z
             tdf.loc[i, "sw"] = (1-alpha)*np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]**alpha
             tdf.loc[i, "sr"] = alpha*np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]**(alpha-1)
             next_k =  alpha * beta * np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]
             tdf.loc[i, "sc"] = tdf.loc[i, "sw"] + tdf.loc[i, "sr"]*tdf.loc[i, "sk"]  - next_k
             tdf.loc[i, "sy"] = np.exp(tdf.loc[i, "z"])*tdf.loc[i, "sk"]**alpha
        
        # model moments
        mmDF.loc[set_num, "em1"] = (np.mean(tdf.sc)-dm1)/dm1
        mmDF.loc[set_num, "em2"] = (np.mean(tdf.sk)-dm2)/dm2
        mmDF.loc[set_num, "em3"] = (np.mean(tdf.sc/tdf.sy)-dm3)/dm3
        mmDF.loc[set_num, "em4"] = (np.var(tdf.sy)-dm4)/dm4
        temp = tdf.copy()
        for i in range(1, len(tdf)):
            temp.loc[i,"prev_c"] = temp.loc[i-1,"sc"]
        temp = temp.loc[:, ["sc", "prev_c"]].copy().dropna()
        mmDF.loc[set_num, "em5"] = (np.corrcoef(temp.sc, temp.prev_c)[0,1]-dm5)/dm5
        temp = tdf.loc[:, ["sc", "sr"]].copy().dropna()
        mmDF.loc[set_num, "em6"] =  (np.corrcoef(temp.sc, temp.sr)[0,1]-dm6)/dm6

    return np.array(mmDF).T

err_mat = err_matrix(df, drawsDF, alpha_GMM, rho_GMM, mu_GMM, sigma_GMM)
omega2 = 1/1000*err_mat @ err_mat.T
w_2step = lin.pinv(omega2)

alpha_init = alpha_GMM
rho_init = rho_GMM
mu_init = mu_GMM
sigma_init = sigma_GMM
params_init = np.array([alpha_init, rho_init, mu_init, sigma_init])
gmm_args = [df, w_2step, drawsDF, False]
results_2step = opt.minimize(criterion, 
    params_init, 
    args=gmm_args, 
    tol=1e-10,
    method='L-BFGS-B', 
    bounds=((0.01, 0.99), (-0.99,0.99), (5,14), (0.01, 1.1)))
results_2step


alpha_2step, rho_2step, mu_2step, sigma_2step = results_2step.x
print("\n alpha_init = {}, rho_init = {}, and mu_init = {}".format(params_init[0], params_init[1],params_init[2],params_init[3]))
print("\n alpha_GMM = {}, rho_GMM = {}, and mu_GMM = {}".format(alpha_2step, rho_2step, mu_2step, sigma_2step))
print("\nValue of Minimized Criterion: {}".format(criterion((alpha_2step, rho_2step, mu_2step, sigma_2step), gmm_args)))

d_err_2step = jacobian_err(df, drawsDF, alpha_2step, rho_2step, mu_2step, sigma_2step, simple = False)

SigHat2_2step = (1 / 1000) * lin.inv(d_err.T_2step @ w_2step) @ d_err_2step)
print(SigHat2_2step)
print('Std. err. alpha_hat=', np.sqrt(SigHat2_2step[0, 0]))
print('Std. err. rho_hat=', np.sqrt(SigHat2_2step[1, 1]))
print('Std. err. mu_hat=', np.sqrt(SigHat2_2step[2, 2]))
print('Std. err. sigma_hat=', np.sqrt(SigHat2_2step[3, 3]))



