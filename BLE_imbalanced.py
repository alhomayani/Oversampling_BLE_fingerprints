import os
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.datasets import make_imbalance
from sklearn.utils import shuffle

os.chdir('/content/gdrive/My Drive/training_testing_data/')

train = pd.read_csv('train_data_rp_3.csv')
X_train = train.iloc[:, :-1]
X_train = X_train.values
Y_train = train.iloc[:, -1:]
Y_train = Y_train.values
Y_train = Y_train.reshape((Y_train.shape[0],))

sampling_strategy={0: 68, 4: 68, 5: 68}
X_train, Y_train = make_imbalance(X_train, Y_train, sampling_strategy=sampling_strategy, random_state=42)
print("IMBALANCED: ",sorted(Counter(Y_train).items()))

Y_train = Y_train.reshape((Y_train.shape[0],1))
train = np.hstack([X_train, Y_train])
train = shuffle(train)
train = pd.DataFrame(train)
train.to_csv('train_data_rp_3_IMBALANCED.csv', index=False)
print("DONE...")