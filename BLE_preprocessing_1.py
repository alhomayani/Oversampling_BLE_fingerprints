import pandas as pd
import os
import numpy as np

# dataset link: http://doi.org/10.5281/zenodo.3927588

deployment = 1

os.chdir('C:/Users/uf11/Desktop/BLE_dataset/Deployment_1')
df_code = pd.read_csv('set1_code.csv')
df_code.columns = ['deployment', 'phone', 'RP', 'beacon', 'slot']
df_rss = pd.read_csv('set1_rss.csv')
df_rss.columns = ['rss']

for slot in range(6):
    for phone in range(3):
        result = pd.DataFrame()
        for RP in range(173):
            result_temp = pd.DataFrame()
            for beacon in range(30):
                df_code_temp = df_code.loc[(df_code['deployment'] == deployment ) & (df_code['phone'] == phone+1) &
                                           (df_code['RP'] == RP+1) & (df_code['beacon'] == beacon+1) &
                                           (df_code['slot'] == slot+1)]
                df_rss_temp = df_rss[df_rss.index.isin(df_code_temp.index)].reset_index(drop=True)
                result_temp = pd.merge(result_temp, df_rss_temp, left_index=True, right_index=True, how='outer')

            result_temp['RP'] = ([int(RP+1)] * len(result_temp))
            result = result.append(result_temp, ignore_index=True)

        result.columns = ['rss_1', 'rss_2', 'rss_3', 'rss_4', 'rss_5','rss_6', 'rss_7', 'rss_8', 'rss_9', 'rss_10', 'rss_11',
                          'rss_12', 'rss_13', 'rss_14', 'rss_15', 'rss_16', 'rss_17', 'rss_18', 'rss_19', 'rss_20', 'rss_21',
                          'rss_22', 'rss_23', 'rss_24', 'rss_25', 'rss_26', 'rss_27', 'rss_28', 'rss_29', 'rss_30', 'RP']

        result = result.replace(-200, np.nan)
        os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_1')
        result.to_csv('phone_'+str(phone+1)+'_slot_'+str(slot+1)+'.csv')
