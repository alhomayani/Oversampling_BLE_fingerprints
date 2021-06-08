# The following code omits RPs that are outside the six zones

import pandas as pd
import os


def rectify_files(d_num):

    os.chdir('C:/Users/uf11/Desktop/BLE_dataset/Deployment_'+str(d_num))
    df_cdr = pd.read_csv('set'+str(d_num)+'_cdr.csv')
    df_cdr.columns = ['x', 'y', 'z']
    df_code = pd.read_csv('set' + str(d_num) + '_code.csv')
    df_code.columns = ['deployment', 'phone', 'RP', 'beacon', 'slot']
    # df_rss = pd.read_csv('set' + str(d_num) + '_rss.csv')
    # df_rss.columns = ['RSS']
    # df_tms = pd.read_csv('set' + str(d_num) + '_tms.csv')
    # df_tms.columns = ['timestamp']


    os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')
    df_reference = pd.read_csv('set'+str(d_num)+'_cdr_NEW.csv')
    df_reference.set_index(df_reference.iloc[:, 0], inplace=True)

    df_cdr = df_cdr[df_cdr.index.isin(df_reference.index)]
    df_cdr.to_csv('set'+str(d_num)+'_cdr_new.csv')
    df_code = df_code[df_code.index.isin(df_reference.index)]
    df_code.to_csv('set' + str(d_num) + '_code_new.csv')
    # df_rss = df_rss[df_rss.index.isin(df_reference.index)]
    # df_rss.to_csv('set' + str(d_num) + '_rss_new.csv')
    # df_tms = df_tms[df_tms.index.isin(df_reference.index)]
    # df_tms.to_csv('set' + str(d_num) + '_tms.csv')


deployment_number = 1 # chose 1 for Deployment_1 and 2 for Deployment_2
rectify_files(deployment_number)