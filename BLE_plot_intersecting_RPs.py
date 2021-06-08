# The following code plots intersecting RPs of Deployment_1 and Deployment_2

import pandas as pd
import os
import matplotlib.pyplot as plt


os.chdir('C:/Users/uf11/Desktop/UEx multi-slotindooroutdoor DB/data/Deployment_1')
df1 = pd.read_csv('set1_cdr.csv')
df1.columns = ['x', 'y', 'z']
df1 = df1.drop_duplicates()
print(len(df1))

os.chdir('C:/Users/uf11/Desktop/UEx multi-slotindooroutdoor DB/data/Deployment_2')
df2 = pd.read_csv('set2_cdr.csv')
df2.columns = ['x', 'y', 'z']
df2 = df2.drop_duplicates()
print(len(df2))

df3 = df2[~df2.isin(df1)].dropna()

df1.drop(df1[((df1['x'] >= 25.5) & (df1['x'] <= 41.0)) | (df1['x'] <= 0.0) | (df1['x'] >= 66.5)].index, inplace=True)
df1.drop(df1[(df1['y'] <= 0.0) | (df1['y'] >= 25.0)].index, inplace=True)
df1.drop(df1[((df1['z'] >= 0.5) & (df1['z'] <= 2.5)) | ((df1['z'] >= 4.5) & (df1['z'] <= 6.5))].index, inplace=True)

df3.drop(df3[((df3['x'] >= 25.5) & (df3['x'] <= 41.0)) | (df3['x'] <= 0.0) | (df3['x'] >= 66.5)].index, inplace=True)
df3.drop(df3[(df3['y'] <= 0.0) | (df3['y'] >= 25.0)].index, inplace=True)
df3.drop(df3[((df3['z'] >= 0.5) & (df3['z'] <= 2.5)) | ((df3['z'] >= 4.5) & (df3['z'] <= 6.5))].index, inplace=True)
print(len(df3))

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(df1.x, df1.y, df1.z, color = "red")
ax.scatter3D(df3.x, df3.y, df3.z, color = "green")

plt.show()

