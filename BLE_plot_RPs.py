# The following code plots the unique RPs

import pandas as pd
import os
import matplotlib.pyplot as plt

def plot_RPs(d_num):

    os.chdir('C:/Users/uf11/Desktop/UEx multi-slotindooroutdoor DB/data/Deployment_'+str(d_num))
    df = pd.read_csv('set'+str(d_num)+'_cdr.csv')
    df.columns = ['x', 'y', 'z']
    df = df.drop_duplicates()

    df.drop(df[((df['x'] >= 25.5) & (df['x'] <= 41.0)) | (df['x'] <=0.0) | (df['x'] >=66.5)].index , inplace=True)
    df.drop(df[(df['y'] <= 0.0) | (df['y'] >= 25.0)].index , inplace=True)
    df.drop(df[((df['z'] >= 0.5) & (df['z'] <= 2.5)) | ((df['z'] >= 4.5) & (df['z'] <= 6.5))].index , inplace=True)

    print('there are (',len(df),') unique RPs')
    plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(df.x, df.y, df.z, color = "green")
    plt.title('RPs of deployment '+str(d_num))
    plt.show()

deployment_number = 1 # chose 1 for Deployment_1 and 2 for Deployment_2
plot_RPs(deployment_number)