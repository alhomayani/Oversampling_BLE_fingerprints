import pandas as pd
import os

result = pd.DataFrame()
os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_2')
for phone in range(1):
    for slot in range(6):
        df_temp = pd.read_csv('phone_'+str(phone+3)+'_slot_'+str(slot+1)+'_scaled.csv', index_col=[0])
        print(len(df_temp))
        result = result.append(df_temp, ignore_index=True)

os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')
result.to_csv('data_3.csv', index=False)
print(len(result))





