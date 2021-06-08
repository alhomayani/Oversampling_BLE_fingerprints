import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


columns = ['rss_1', 'rss_2', 'rss_3', 'rss_4', 'rss_5','rss_6', 'rss_7', 'rss_8', 'rss_9', 'rss_10', 'rss_11',
           'rss_12', 'rss_13', 'rss_14', 'rss_15', 'rss_16', 'rss_17', 'rss_18', 'rss_19', 'rss_20', 'rss_21',
           'rss_22', 'rss_23', 'rss_24', 'rss_25', 'rss_26', 'rss_27', 'rss_28', 'rss_29', 'rss_30']


for phone in range(3):
    for slot in range(6):
        os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_1')
        df = pd.read_csv('phone_'+str(phone+1)+'_slot_'+str(slot+1)+'.csv')
        RP = df[['RP']]
        df = df[columns]
        x = df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(x)
        x_scaled = scaler.transform(x)
        x_scaled = np.nan_to_num(x_scaled)
        df_scaled = pd.DataFrame(x_scaled)
        os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_2')
        df_scaled.columns = columns
        df_scaled['RP'] = RP
        df_scaled.to_csv('phone_'+str(phone+1)+'_slot_'+str(slot+1)+'_scaled.csv')


