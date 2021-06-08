# filter out RPs and then assign zones
# divide into training and testing


import pandas as pd
import os


os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')

df1 = pd.read_csv('set1_code_new.csv', index_col=[0])

df2 = pd.read_csv('set1_cdr_new.csv', index_col=[0])

result = pd.concat([df1, df2], axis=1)

result = result[['RP','x','y','z']]

print(len(result))

result = result.drop_duplicates()

result.to_csv('RP_cdr.csv')

print(len(result))







