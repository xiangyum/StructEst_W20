
import math 
import scipy.optimize as opt
import numpy as np
import pandas as pd

df = pd.read_csv("clms.txt", header=None)
df["x"] = df[0]

def gg_pdf(df, beta_v, alpha_v, m_v):
    df["pdf_values"] = m_v/(beta_v**alpha_v*math.gamma(alpha_v/m_v))*df.x**(alpha_v-1)*np.exp(-1*(df.x/beta_v)**m_v)
    df.loc[df["pdf_values"] < 1e-7, "pdf_values"] = 1e-7
    return np.log(df["pdf_values"])

def crit_gg(params, *args):
    beta_value, alpha_value, m_value = params
    df = list(args)[0]
    df_loglik = gg_pdf(df, beta_value, alpha_value, m_value)
    negloglik = -sum(df_loglik)
    return negloglik    

params_init = np.array([beta_MLE, alpha_MLE, 1])
mle_args = (df)
results_uncstr_gg = opt.minimize(crit_gg, params_init, args=(mle_args),  method='TNC',
                 bounds=((0.01, None), (0.010, None), (0.01, None)), tol=1e-3, options={'maxiter':1000})


########################################
########################################
########################################
########################################
########################################


def gb2_pdf(df, a_value, b_value, p_value, q_value):
    df["pdf_values"] = a_value*df.x**(a_value*p_value-1)/b_value**(a_value*p_value)/beta(p_value, q_value)/(1+(df.x/b_value)**a_value)**(p_value+ q_value)
    df.loc[df["pdf_values"] < 1e-15, "pdf_values"] = 1e-15
    return np.log(df["pdf_values"])

def crit_gb2(params, *args):
    a_value, b_value, p_value, q_value = params
    df = list(args)[0]
    df_loglik = np.log(gb2_pdf(df, a_value, b_value, p_value, q_value))
    negloglik = -sum(df_loglik)
    return negloglik

params_init = np.array([beta_MLE_gg, alpha_MLE_gg, m_MLE_gg, 10000])
beta_value, alpha_value, m_value, q_value = params_init
a_value = m_value
b_value = q_value**(1/m_value)*beta_value
p_value = alpha_value/m_value
params_init2 =  np.array([a_value, b_value, p_value, q_value])
mle_args = (df)
results_uncstr_gb2 = opt.minimize(crit_gb2, params_init2, args=(mle_args),  method='TNC',
                 bounds=((1e-2, None), (1e-2, None), (1e-2, None), (1e-2, None)), tol=1e-5, options={'maxiter':3000})


def gb2_pdf_plotting(df, a_value, b_value, p_value, q_value):
    df = pd.DataFrame({'x':df})
    df["pdf_values"] = a_value*df.x**(a_value*p_value-1)/b_value**(a_value*p_value)/beta(p_value, q_value)/(1+(df.x/b_value)**a_value)**(p_value+ q_value)
    df.loc[df["pdf_values"] < 1e-3, "pdf_values"] = 1e-3
    return df["pdf_values"].tolist()



a_value, b_value, p_value, q_value = params_init
testdf = df.head()

a_value*testdf.x**(a_value*p_value-1)/b_value**(a_value*p_value)/beta(p_value, q_value)/(1+(testdf.x/b_value)**a_value)**(p_value+ q_value)


########################################
########################################
########################################
########################################
########################################


def q2_pdf1(df, alpha_v, rho_v, mu_v, sigma_v):
    df["prob"] = np.nan
    df["z"] = np.nan
    for index, row in df.iterrows():
        if index == 0:
            prev_z = mu_v
            current_z = np.log(row.w/(1-alpha_v)/row.k**alpha_v)
            current_mean = (rho_v*prev_z) + (1- rho_v)*mu_v
            df.loc[index, "z"] = current_z
            prob_t = 1/sigma_v/math.sqrt(2*math.pi)*np.exp(-0.5*((current_z- current_mean)/sigma_v)**2)
            if prob_t < 1e-8:
                prob_t = 1e-8
            df.loc[index, "prob"] = prob_t
            
        else:
            t_prev = index -1
            prev_z = df.loc[t_prev, "z"]
            current_z = np.log(row.w/(1-alpha_v)/row.k**alpha_v)
            current_mean = (rho_v*prev_z) + (1- rho_v)*mu_v
            df.loc[index, "z"] = current_z
            prob_t = 1/sigma_v/math.sqrt(2*math.pi)*np.exp(-0.5*((current_z- current_mean)/sigma_v)**2)
            if prob_t < 1e-8:
                prob_t = 1e-8
            df.loc[index, "prob"] = prob_t
    return df["prob"]



def crit_q2pdf1(params, *args):
    alpha_v, rho_v, mu_v, sigma_v = params
    df = list(args)[0]
    pdf_values = np.log(q2_pdf1(df, alpha_v, rho_v, mu_v, sigma_v))
    negloglik = -sum(pdf_values)
    return negloglik


params_init = np.array([0.4, 0.4, 1.5, 1.5])
mle_args = (mdf)
results_uncstr_q2 = opt.minimize(crit_q2pdf1, params_init, args=(mle_args),  method='TNC',
    bounds=((1e-10, 1-1e-10), (-1+1e-10, 1-1e-10), (1e-10, None), (None, None)), tol=1e-12)

alpha_q2MLE, rho_q2MLE, mu_q2MLE, sigma_q2MLE = results_uncstr_q2.x


######

def q2_pdf2(df, alpha_v, rho_v, mu_v, sigma_v):
    df["prob"] = np.nan
    df["z"] = np.nan
    for index, row in df.iterrows():
        if index == 0:
            prev_z = mu_v
            current_z = np.log(row.r/alpha_v/row.k**(alpha_v-1))
            current_mean = (rho_v*prev_z) + (1- rho_v)*mu_v
            df.loc[index, "z"] = current_z
            prob_t = 1/sigma_v/math.sqrt(2*math.pi)*np.exp(-0.5*((current_z- current_mean)/sigma_v)**2)
            if prob_t < 1e-8:
                prob_t = 1e-8
            df.loc[index, "prob"] = prob_t
            
        else:
            t_prev = index -1
            prev_z = df.loc[t_prev, "z"]
            current_z = np.log(row.r/alpha_v/row.k**(alpha_v-1))
            current_mean = (rho_v*prev_z) + (1- rho_v)*mu_v
            df.loc[index, "z"] = current_z
            prob_t = 1/sigma_v/math.sqrt(2*math.pi)*np.exp(-0.5*((current_z- current_mean)/sigma_v)**2)
            if prob_t < 1e-8:
                prob_t = 1e-8
            df.loc[index, "prob"] = prob_t
    return df["prob"]



def crit_q2pdf2(params, *args):
    alpha_v, rho_v, mu_v, sigma_v = params
    df = list(args)[0]
    pdf_values = np.log(q2_pdf2(df, alpha_v, rho_v, mu_v, sigma_v))
    negloglik = -sum(pdf_values)
    return negloglik



