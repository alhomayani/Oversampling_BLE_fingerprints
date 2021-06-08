import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
import matplotlib as mpl
import os
import pandas as pd

size = 10
mpl.rc('font', family='serif')
plt.rc('xtick', labelsize=size-1)
plt.rc('ytick', labelsize=size-1)

os.chdir('C:/Users/uf11/Desktop/BLE_dataset/preprocessing/preprocessing_3')

df = pd.read_csv('random_samples.csv')
zones = df[['zone']]

zones = zones.values
df_rss = df.drop(['zone'], axis=1)
npa_rss = df_rss.values

rp = RecurrencePlot()
npa_rss_rp = rp.fit_transform(npa_rss)

for i in range(6):
    plt.subplot(230 + 1 + i)
    plt.title('Symbolic Space %i' %zones[i])
    plt.imshow(npa_rss_rp[i], cmap=plt.get_cmap('gray'))

plt.show()


