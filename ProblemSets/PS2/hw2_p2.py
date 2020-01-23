import pandas as pd
import numpy as np
from scipy.special import beta
from scipy import integrate
import matplotlib.pyplot as plt

# import data:
df = pd.read_csv("./clms.txt", header = None)((asddmomds))
df["x"] = df[0]

# Q1:   

from scipy import integrate

def prob_gb2(x_value):
    a_value = 0.8370464113216801
    b_value = 36324509.41502439
    p_value = 0.8013535696450651
    q_value = 9871.420693943379
    prob = a_value*x_value**(a_value*p_value-1)/b_value**(a_value*p_value)/beta(p_value, q_value)/(1+(x_value/b_value)**a_value)**(p_value + q_value)
    if prob < 1e-15:
        prob = 1e-15
    return prob

def prob_ga(x_value):
    alpha_value = 0.7102143006296369
    beta_value = 551.0179500811464
    prob = 1/(beta_value**(alpha_value) * math.gamma(alpha_value))*x_value**(alpha_value-1)*np.exp(-x_value/beta_value)

    if prob < 1e-15:
        prob = 1e-15
    return prob


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


