import pandas as pd
import os
from math import comb

dir = os.getcwd()

def get_analysis(k_list, n_features, n_samples):

    #priors 
    df_full = pd.read_csv(f"{dir}/analysis/data/exp4/EP_k_{n_features}_Nlat", sep=', ',header=None, engine='python')
    df_full = df_full.transpose()

    for k in k_list:
        df_nlat = pd.read_csv(f"{dir}/analysis/data/exp4/EP_k_{k}_Nlat", sep=', ',header=None, engine='python')
        df_nlat = df_nlat.transpose()
        
        # df_lat = pd.read_csv(f"{dir}/analysis/data/exp2/EP_k_{k}_Fair", sep=', ',header=None, engine='python')
        # df_lat = df_lat.transpose()
        
        print("------------------------------------------------------------------")
        print(f'Mean of all EP with k={k} on nonLatent PC    : {df_nlat[0].mean()}')
        print(f'StdDev of all EP with k={k} on nonLatent PC  : {df_nlat[0].std()}')
        # print(f'mean of all EP with k={k} on FairPC          : {df_lat[0].mean()}')
        # print(f'StdDev of all EP with k={k} on FairPC        : {df_lat[0].std()}')
        print("------------------------------------------------------------------")
        
        instances = pd.read_csv(f'{dir}/analysis/data/exp4/sampled_instances_{n_samples}.csv')

        vals = pd.DataFrame(columns=['instance','nlat mean','nlat stdDev', 'prior'],index=range(n_samples))
        for idx in range(n_samples):
            t1 = df_nlat.iloc[(comb(n_features,k)*idx):comb(n_features,k)*(idx+1)-1]
            vals.loc[idx] = [list(instances.iloc[idx][:]),t1[0].mean(),t1[0].std(), df_full.loc[idx][0]]

        vals.to_csv(f'{dir}/analysis/data/exp4/{k}_analysis.csv', index=False)

get_analysis([7,8,15,16,21],21,10)