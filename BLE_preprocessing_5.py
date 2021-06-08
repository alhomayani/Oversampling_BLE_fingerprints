import pandas as pd
import os


os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')

df = pd.read_csv('RP_cdr.csv', index_col=[0])
result = pd.DataFrame()

z1 = df[(df['x'] < 30.0) & (df['z'] < 2)]
z1['zone'] = ([int(0)] * len(z1))
result = result.append(z1, ignore_index=True)
z2 = df[(df['x'] > 30.0) & (df['z'] < 2)]
z2['zone'] = ([int(1)] * len(z2))
result = result.append(z2, ignore_index=True)
z3 = df[(df['x'] < 30.0) & ((df['z'] > 2) &(df['z'] < 5))]
z3['zone'] = ([int(2)] * len(z3))
result = result.append(z3, ignore_index=True)
z4 = df[(df['x'] > 30.0) & ((df['z'] > 2) &(df['z'] < 5))]
z4['zone'] = ([int(3)] * len(z4))
result = result.append(z4, ignore_index=True)
z5 = df[(df['x'] < 30.0) & (df['z'] > 5)]
z5['zone'] = ([int(4)] * len(z5))
result = result.append(z5, ignore_index=True)
z6 = df[(df['x'] > 30.0) & (df['z'] > 5)]
z6['zone'] = ([int(5)] * len(z6))
result = result.append(z6, ignore_index=True)

result.to_csv('RP_cdr_zone.csv', index=False)






