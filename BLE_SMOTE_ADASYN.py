import os
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.utils import shuffle


os.chdir('/content/gdrive/My Drive/training_testing_data/')

train = pd.read_csv('train_data_rp_3_IMBALANCED.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values

oversample = SMOTE()
X_train_SMOTE, Y_train_SMOTE = oversample.fit_resample(X_train, Y_train)
print('SMOTE:', sorted(Counter(Y_train_SMOTE).items()))
X_train_SMOTE, Y_train_SMOTE = shuffle(X_train_SMOTE, Y_train_SMOTE, random_state=42)
X_train_SMOTE = pd.DataFrame(X_train_SMOTE)
Y_train_SMOTE = pd.DataFrame(Y_train_SMOTE)
train_SMOTE = pd.concat([X_train_SMOTE, Y_train_SMOTE], axis=1, ignore_index=True)
train_SMOTE.to_csv('train_data_rp_3_SMOTE.csv', index=False)

oversample = ADASYN()
X_train_ADASYN, Y_train_ADASYN = oversample.fit_resample(X_train, Y_train)
print('ADASYN:', sorted(Counter(Y_train_ADASYN).items()))
X_train_ADASYN, Y_train_ADASYN = shuffle(X_train_ADASYN, Y_train_ADASYN, random_state=42)
X_train_ADASYN = pd.DataFrame(X_train_ADASYN)
Y_train_ADASYN = pd.DataFrame(Y_train_ADASYN)
train_ADASYN = pd.concat([X_train_ADASYN, Y_train_ADASYN], axis=1, ignore_index=True)
train_ADASYN.to_csv('train_data_rp_3_ADASYN.csv', index=False)

print("DONE...")