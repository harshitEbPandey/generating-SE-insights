import pandas as pd
import os
from math import comb

dir = os.getcwd()

def get_analysis(k_list, n_samples):
    for k in k_list:
        df_nlat = pd.read_csv(f"{dir}/analysis/data/EP_k_{k}_Nlat", sep=', ',header=None, engine='python')
        df_nlat = df_nlat.transpose()
        
        df_lat = pd.read_csv(f"{dir}/analysis/data/EP_k_{k}_Fair", sep=', ',header=None, engine='python')
        df_lat = df_lat.transpose()
        
        print("------------------------------------------------------------------")
        print(f'Mean of all EP with k={k} on nonLatent PC    : {df_nlat[0].mean()}')
        print(f'StdDev of all EP with k={k} on nonLatent PC  : {df_nlat[0].std()}')
        print(f'mean of all EP with k={k} on FairPC          : {df_lat[0].mean()}')
        print(f'StdDev of all EP with k={k} on FairPC        : {df_lat[0].std()}')
        print("------------------------------------------------------------------")
        
        instances = pd.read_csv(f'{dir}/analysis/data/sampled_instances_{n_samples}.csv')

        vals = pd.DataFrame(columns=['instance','nlat mean','nlat stdDev', 'fair mean' , 'fair stdDev'],index=range(comb(7,k)))
        for idx in range(comb(7,k)):
            t1 = df_nlat.iloc[(35*idx):35*(idx+1)-1]
            t2 = df_lat.iloc[(35*idx):35*(idx+1)-1]
            vals.loc[idx] = [list(instances.iloc[idx][:]),t1[0].mean(),t1[0].std(),t2[0].mean(),t2[0].std()]

        vals.to_csv(f'{dir}/analysis/data/{k}_analysis.csv', index=False)

get_analysis([3,4,5],50)