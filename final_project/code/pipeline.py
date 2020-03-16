import pandas as pd
import numpy as np
from sympy.solvers import solve
from sympy import Symbol

def prepare_data(fn):
    # reads a fn
    # outputs a tuple containing df, mdf, and wdf
    df = pd.read_csv(fn)
    df["married_status"] = 0
    df.loc[df.year_ended > 1993, "married_status"] = 1.0
    df["t_single"] = df["age_ms"] - 15
    df["t_married"] = df["age_me"] -  df["age_ms"]    

    df.loc[df.t_married > 25,"t_married"]= 25

    df = df.loc[df.t_married >0,:].copy()
    df['z_raw'] = np.exp(df.wage_ms)
    # df["z"] = pd.qcut(df.z_raw, q = 7, duplicates= "drop", labels =[1,2,3,4,5,6,7])
    mdf = df.loc[df.gender ==1, :].copy().reset_index()
    wdf = df.loc[df.gender ==0, :].copy().reset_index()
    mdf["z"], bins_mz = pd.qcut(mdf.z_raw, q = 5, labels =[1,2,3,4,5], retbins=True)
    wdf["z"], bins_wz = pd.qcut(wdf.z_raw, q = 5, labels =[1,2,3,4,5], retbins=True)

    mdf["x"] = np.nan
    mdf.loc[mdf.z == 1, "x"] = np.median(mdf.loc[mdf.z==1, "z_raw"])
    mdf.loc[mdf.z == 2, "x"] = np.median(mdf.loc[mdf.z==2, "z_raw"])
    mdf.loc[mdf.z == 3, "x"] = np.median(mdf.loc[mdf.z==3, "z_raw"])
    mdf.loc[mdf.z == 4, "x"] = np.median(mdf.loc[mdf.z==4, "z_raw"])
    mdf.loc[mdf.z == 5, "x"] = np.median(mdf.loc[mdf.z==5, "z_raw"])

    wdf["x"] = np.nan
    wdf.loc[wdf.z == 1, "x"] = np.median(wdf.loc[wdf.z==1, "z_raw"])
    wdf.loc[wdf.z == 2, "x"] = np.median(wdf.loc[wdf.z==2, "z_raw"])
    wdf.loc[wdf.z == 3, "x"] = np.median(wdf.loc[wdf.z==3, "z_raw"])
    wdf.loc[wdf.z == 4, "x"] = np.median(wdf.loc[wdf.z==4, "z_raw"])
    wdf.loc[wdf.z == 5, "x"] = np.median(wdf.loc[wdf.z==5, "z_raw"])

    # assign partner's z & x

    partnerDF = pd.DataFrame({"i_type": list(wdf["z"].drop_duplicates().sort_values()),
                             "x": list(wdf["x"].drop_duplicates().sort_values())})
    for index, row in partnerDF.iterrows():
        lb = bins_wz[index]
        ub = bins_wz[index + 1]
        if index == 0:
            lb = -0.01
        partnerDF.loc[index, "lb"] = lb
        partnerDF.loc[index, "ub"] = ub

    mdf["partner_z"] = np.nan
    mdf["partner_x"] = np.nan

    for index, row in mdf.iterrows():
        temp_wage = row["partnerwage_ms"]
        temp_z = np.exp(temp_wage)
        temp = partnerDF.loc[(partnerDF.lb < temp_z) & (partnerDF.ub >= temp_z),["i_type", 'x']]
        try:
            mdf.loc[index, "partner_z"] = np.float(temp.i_type)
        except:
            pass
        try:
            mdf.loc[index, "partner_x"] = np.float(temp.x)
        except: 
            pass


    return (df, mdf, wdf)

def calc_mrhs(wdf, lb, ub):
    acceptance_set = [i for i in range(1,6) if (i >= lb) & (i <= ub)]
    big_sum = 0
    ri = np.median(wdf.loc[wdf.z == lb, "x"])
    for xj in acceptance_set:
        big_sum = big_sum + 0.2*((np.median(wdf.loc[wdf.z == xj, "x"]))-ri)
    rhs = 2 + (est_lambda)/(est_beta + est_delta)*big_sum
    return(rhs)

def find_reserve(df_opp, ub, params):
    est_lambda, est_delta = params
    est_beta = 0.05

    diff_df = pd.DataFrame({'r' : [i for i in range(1,ub + 1)],
        "lhs": np.nan,
        "rhs": np.nan,
        "delta": np.nan})

    for i in range(1,ub + 1):
        lb = ub + 1-i
        acceptance_set = [i for i in range(1,6) if (i >= lb) & (i <= ub)]
        big_sum = 0
        ri = np.median(df_opp.loc[df_opp.z == lb, "x"])
        diff_df.loc[diff_df.r == lb, "lhs"] = ri 
        for xj in acceptance_set:
            big_sum = big_sum + 0.2*((np.median(df_opp.loc[df_opp.z == xj, "x"]))-ri)
        rhs = 2 + (est_lambda)/(est_beta + est_delta)*big_sum
        diff_df.loc[diff_df.r == lb, "rhs"] = rhs 
        delta = ri-rhs
        diff_df.loc[diff_df.r == lb, "delta"] = delta 

    diff_df["abs_delta"] = np.abs(diff_df.delta)
    chosen_type = diff_df.sort_values("abs_delta").reset_index(drop = True).loc[0,"r"]
    # chosen_type = diff_df.loc[diff_df.delta >0, :].sort_values("delta").reset_index(drop = True).loc[0,"r"]
    return(chosen_type)

def set_max_alt(df_opp):
    
    try:
        ub = df_opp.loc[df_opp.reserv.isnull(),:].sort_values("type", ascending = False).reset_index(drop = True).loc[0, "type"]
    except:
        ub = 1
        pass
    return(ub)

def reserve_alt(df_main, df_opp, main_type, params):
    # set max of
    ub = df_opp.loc[df_opp.reserv.isnull(),:].sort_values("type", ascending = False).reset_index(drop = True).loc[0, "type"]
    # retrieve highest empty spot, that will be new max
    diff_df = pd.DataFrame({'r' : [i for i in range(1,ub + 1)],
            "lhs": np.nan,
            "rhs": np.nan,
            "delta": np.nan,})
    for i in range(1,ub + 1):
        lb = ub + 1-i
        acceptance_set = [i for i in range(1,6) if (i >= lb) & (i <= ub)]
        big_sum = 0
        ri = np.median(df_opp.loc[df_opp.z == lb, "x"])
        diff_df.loc[diff_df.r == lb, "lhs"] = ri 
        for xj in acceptance_set:
            big_sum = big_sum + 0.2*((np.median(df_opp.loc[df_opp.z == xj, "x"]))-ri)
        rhs = 2 + (est_lambda)/(est_beta + est_delta)*big_sum
        diff_df.loc[diff_df.r == lb, "rhs"] = rhs 
        delta = ri-rhs
        diff_df.loc[diff_df.r == lb, "delta"] = delta 
    diff_df["abs_delta"] = np.abs(diff_df.delta)
    chosen_type = diff_df.sort_values("abs_delta").reset_index(drop = True).loc[0,"r"]
    return(chosen_type)

def generate_matchsets(mdf, wdf, params):
    est_lambda, est_delta = params
    est_beta = 0.05

    menDF= pd.DataFrame({'type' : [6-i for i in range(1,6)],
                         "max": np.nan,
                         "reserv": np.nan})
    womenDF = pd.DataFrame({'type' : [6-i for i in range(1,6)],
                         "max": np.nan,
                                  "reserv": np.nan})

    # set max for top categories of men and women
    menDF.loc[menDF.type == 5, "max"] = 5
    womenDF.loc[womenDF.type == 5, "max"] = 5
    menDF.loc[menDF.type == 5, "reserv"] = find_reserve(wdf, 5, params)
    womenDF.loc[womenDF.type == 5, "reserv"] = find_reserve(mdf, 5, params)

    # for men
    for i in range(1,5):
        i_type = 5-i
        
        man_max_exists = False
        
        # set max for men if identifiable
        try:
            max_men_i = list(womenDF.loc[womenDF.reserv <= i_type, "type"].sort_values(ascending = False))[0]
            man_max_exists = True

        except:
            pass
        
        # set reserve for men if identifiable 
        if man_max_exists == True:
            # set max:
            menDF.loc[menDF.type == i_type, "max"] = max_men_i
            #print("Max of men of type {} = {}".format(i_type, max_men_i))
            menDF.loc[menDF.type == i_type, "reserv"] = find_reserve(wdf, max_men_i, params)
            #print("Reserve of men of type {} = {}".format(i_type, find_reserve(wdf, max_men_i, est_params)))
  
        # set max for max if unidentifable:
        
        if man_max_exists == False:
            new_man_max = set_max_alt(womenDF)
            menDF.loc[menDF.type ==i_type, "max"] = new_man_max
            #print("Max of men of type {} = {}".format(i_type, new_man_max))
            menDF.loc[menDF.type == i_type, "reserv"] = find_reserve(wdf, new_man_max, params)
            #print("Reserve of men of type {} = {}".format(i_type, find_reserve(wdf, new_man_max, est_params)))


        # set max for women if identifiable
        woman_max_exists = False
        
        try:
                max_women_i = list(menDF.loc[menDF.reserv <= i_type, "type"].sort_values(ascending = False))[0]
                woman_max_exists = True
        except:
            pass

        if woman_max_exists == True:
            # set max:
            womenDF.loc[womenDF.type == i_type, "max"] = max_women_i
            #print("Max of women of type {} = {}".format(i_type, max_women_i))
            womenDF.loc[womenDF.type == i_type, "reserv"] = find_reserve(mdf, max_women_i, params)
            #print("Reserve of women of type {} = {}".format(i_type, find_reserve(mdf, max_women_i, est_params)))
        
        if woman_max_exists == False:
            new_woman_max = set_max_alt(menDF)
            womenDF.loc[womenDF.type ==i_type, "max"] = new_woman_max
            #print("Max of women of type {} = {}".format(i_type, new_woman_max))
            womenDF.loc[womenDF.type == i_type, "reserv"] = find_reserve(mdf, new_woman_max, params)
            #print("Reserve of women of type {} = {}".format(i_type, find_reserve(mdf, new_woman_max, est_params)))

    # go back to revise maxes:

    for i in range(1,6):
        i_type = 6-i
        # check for men:
        real_max = max(womenDF.loc[womenDF.reserv <= i_type, "type"].sort_values(ascending = False))
        actual_max = np.float(menDF.loc[menDF.type == i_type, "max"])
        if real_max > actual_max:
            menDF.loc[menDF.type == i_type, "max"] = real_max
        
        # check for women:
        try:
            real_max = max(menDF.loc[menDF.reserv <= i_type, "type"].sort_values(ascending = False))
            actual_max = np.float(womenDF.loc[womenDF.type == i_type, "max"])
            if real_max > actual_max:
                womenDF.loc[womenDF.type == i_type, "max"] = real_max
        except: 
            pass
    return (menDF, womenDF)


def generate_likelihoods(df, mdf, wdf, params):
    # unpack parameters
    est_lambda, est_delta = params
    est_beta = 0.05

    match_sets = generate_matchsets(mdf, wdf, params)
    match_men, match_women =  match_sets[0], match_sets[1]

    mdf["lik_single"] = np.nan
    mdf["lik_married"] = np.nan

    for index, row in mdf.iterrows():
        # likelihood of being single:
        temp_z = row.z
        temp_max = np.float(match_men.loc[match_men.type == row.z, "max"])
        temp_reserv = np.float(match_men.loc[match_men.type == row.z, "reserv"])
        single_time = row.t_single
        married_time = row.t_married
        temp_gamma = 0.2 * (1 + temp_max - temp_reserv)
        likelihood_single = est_lambda * temp_gamma * np.exp(-est_lambda*temp_gamma*single_time)

        # likelihood of marriage
        fxj = 1/(1 + temp_max - temp_reserv)
        likelihood_married = fxj * est_delta * np.exp(-est_delta * married_time)
        mdf.loc[index, "lik_single"] = likelihood_single
        mdf.loc[index, "lik_married"] = likelihood_married

    mdf["likelihood"] =  mdf["lik_single"] * mdf["lik_married"] 
    return mdf["likelihood"]

def criterion_function(params, args):
    df = args[0]
    mdf = args[1]
    wdf = args[2]
    likelihood_vector = generate_likelihoods(df, mdf, wdf, params)
    negloglik = -sum(np.log(likelihood_vector))
    return negloglik













