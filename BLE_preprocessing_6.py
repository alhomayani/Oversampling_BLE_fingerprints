import pandas as pd
import os


os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')

df_RP_cdr_zone = pd.read_csv('RP_cdr_zone.csv', index_col=[0])
df_data = pd.read_csv('data_3.csv')

result = pd.merge(df_data, df_RP_cdr_zone, how='inner',left_on=['RP'], right_on=['RP'])
result = result.drop(['RP', 'x','y','z'], axis=1)

total_samples = 8500

z1_data = result[result['zone'] == 0]
z1_data = z1_data.sample(frac=1).reset_index(drop=True)
print(len(z1_data))
z1_data = z1_data.head(total_samples)

z2_data = result[result['zone'] == 1]
z2_data = z2_data.sample(frac=1).reset_index(drop=True)
print(len(z2_data))
z2_data = z2_data.head(total_samples)

z3_data = result[result['zone'] == 2]
z3_data = z3_data.sample(frac=1).reset_index(drop=True)
print(len(z3_data))
z3_data = z3_data.head(total_samples)

z4_data = result[result['zone'] == 3]
z4_data = z4_data.sample(frac=1).reset_index(drop=True)
print(len(z4_data))
z4_data = z4_data.head(total_samples)

z5_data = result[result['zone'] == 4]
z5_data = z5_data.sample(frac=1).reset_index(drop=True)
print(len(z5_data))
z5_data = z5_data.head(total_samples)

z6_data = result[result['zone'] == 5]
z6_data = z6_data.sample(frac=1).reset_index(drop=True)
print(len(z6_data))
z6_data = z6_data.head(total_samples)

train_data = pd.DataFrame()
test_data = pd.DataFrame()

train_samples = 6800
test_samples = 1700
z1_train = z1_data.head(train_samples)
z1_test = z1_data.tail(test_samples)
z2_train = z2_data.head(train_samples)
z2_test = z2_data.tail(test_samples)
z3_train = z3_data.head(train_samples)
z3_test = z3_data.tail(test_samples)
z4_train = z4_data.head(train_samples)
z4_test = z4_data.tail(test_samples)
z5_train = z5_data.head(train_samples)
z5_test = z5_data.tail(test_samples)
z6_train = z6_data.head(train_samples)
z6_test = z6_data.tail(test_samples)

train_data = train_data.append(z1_train, ignore_index=True)
train_data = train_data.append(z2_train, ignore_index=True)
train_data = train_data.append(z3_train, ignore_index=True)
train_data = train_data.append(z4_train, ignore_index=True)
train_data = train_data.append(z5_train, ignore_index=True)
train_data = train_data.append(z6_train, ignore_index=True)

test_data = test_data.append(z1_test, ignore_index=True)
test_data = test_data.append(z2_test, ignore_index=True)
test_data = test_data.append(z3_test, ignore_index=True)
test_data = test_data.append(z4_test, ignore_index=True)
test_data = test_data.append(z5_test, ignore_index=True)
test_data = test_data.append(z6_test, ignore_index=True)

train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)

train_data.to_csv('train_data_3.csv', index=False)
test_data.to_csv('test_data_3.csv', index=False)

