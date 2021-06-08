# The following code omits RPs that are outside the six zones for Deployment_1 and Deployment_2

import pandas as pd
import os


def omit_RPs(d_num):

    os.chdir('C:/Users/uf11/Desktop/BLE_dataset/Deployment_'+str(d_num))
    df = pd.read_csv('set'+str(d_num)+'_cdr.csv')
    df.columns = ['x', 'y', 'z']
    l1 = len(df)
    df.drop(df[((df['x'] >= 25.5) & (df['x'] <= 41.0)) | (df['x'] <=0.0) | (df['x'] >=66.5)].index , inplace=True)
    df.drop(df[(df['y'] <= 0.0) | (df['y'] >= 25.0)].index , inplace=True)
    df.drop(df[((df['z'] >= 0.5) & (df['z'] <= 2.5)) | ((df['z'] >= 4.5) & (df['z'] <= 6.5))].index , inplace=True)
    l2 = len(df)
    os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')
    df.to_csv('set'+str(d_num)+'_cdr_NEW.csv')
    print('percentage of dropped samples: ', (l1-l2)/l1)


deployment_number = 1 # chose 1 for Deployment_1 and 2 for Deployment_2
omit_RPs(deployment_number)