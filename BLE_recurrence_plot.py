import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from pyts.image import GramianAngularField, MarkovTransitionField
from pyts.datasets import load_gunpoint
import os
import pandas as pd
import numpy as np



os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')
phases = ['train', 'test']
for phase in phases:

    df_train = pd.read_csv(phase+'_data_3.csv')
    zones = df_train[['zone']]

    zones = zones.values
    df_rss = df_train.drop(['zone'], axis=1)
    npa_rss = df_rss.values

    rp = RecurrencePlot()
    npa_rss_rp = rp.fit_transform(npa_rss)
    d0 = npa_rss_rp.shape[0]
    d1 = npa_rss_rp.shape[1]

    npa_rss_rp = np.reshape(npa_rss_rp,(d0, d1*d1))

    npa_rss_zone_rp = np.hstack((npa_rss_rp,zones))

    df_rss_zone_rp = pd.DataFrame(npa_rss_zone_rp)

    df_rss_zone_rp.to_csv(phase+'_data_rp_3.csv', index=False)




