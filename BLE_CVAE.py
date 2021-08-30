# code inspired from: R. Atienza,Advanced Deep Learning with Keras: Apply deep learningtechniques,  autoencoders,  GANs,  variational  autoencoders,  deep  rein-forcement learning, policy gradients, and more.

import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D, \
    UpSampling2D, Reshape, concatenate, BatchNormalization
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.losses import mse, binary_crossentropy
import os
import pandas as pd
from keras.models import load_model
from imblearn.datasets import make_imbalance
from sklearn.utils import shuffle
from collections import Counter

def sampling(args):
    mu, l_sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * l_sigma) * epsilon


def construct_numvec(digit, z=None):
    out = np.zeros((1, n_z + n_y))
    out[:, digit + n_z] = 1.
    if z is None:
        return out
    else:
        for i in range(len(z)):
            out[:, i] = z[i]
        return out


os.chdir('/content/gdrive/My Drive/training_testing_data/')

train = pd.read_csv('train_data_rp_3_IMBALANCED.csv')
x_train = train.iloc[:, :-1]
x_train = x_train.values
y_train = train.iloc[:, -1:]
y_train = y_train.values

x_train = x_train.reshape((x_train.shape[0], 30, 30))
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])


num_labels = len(np.unique(y_train))

# network parameters
input_shape = (image_size, image_size, 1)
label_shape = (num_labels,)
batch_size = 64
kernel_size = 4
filters = 8
n_z = 2
epochs = 10000

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
y_labels = Input(shape=label_shape, name='class_labels')
x = Dense(image_size * image_size)(y_labels)
x = Reshape((image_size, image_size, 1))(x)
x = concatenate([inputs, x])

filters *= 2
x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)
filters *= 2
x = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)


# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
mu = Dense(n_z, name='z_mean')(x)
l_sigma = Dense(n_z, name='z_log_var')(x)

z = Lambda(sampling, output_shape=(n_z,), name='z')([mu, l_sigma])

# instantiate encoder model
encoder = Model([inputs, y_labels], [mu, l_sigma, z], name='encoder')
encoder.summary()


# build decoder model
latent_inputs = Input(shape=(n_z,), name='z_sampling')
x = concatenate([latent_inputs, y_labels])
x = Dense(shape[1]*shape[2]*shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)


x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2,  padding='same', output_padding=1)(x)
filters //= 2
x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, activation='relu', strides=2, padding='same')(x)

outputs = Conv2DTranspose(filters=1, kernel_size=kernel_size, activation='sigmoid', padding='same', name='decoder_output')(x)

# instantiate decoder model
decoder = Model([latent_inputs, y_labels], outputs, name='decoder')
decoder.summary()


# instantiate vae model
outputs = decoder([encoder([inputs, y_labels])[2], y_labels])
cvae = Model([inputs, y_labels], outputs, name='cvae')

reconstruction_loss = binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
# reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= image_size * image_size
kl_loss = 1 + l_sigma - K.square(mu) - K.exp(l_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5 * 1
cvae_loss = K.mean(reconstruction_loss + kl_loss)
cvae.add_loss(cvae_loss)
opt = Adam(lr=0.0001)
cvae.compile(optimizer=opt)
cvae.summary()

my_callbacks = [EarlyStopping(patience=100), ModelCheckpoint('model_cvae.h5', monitor='val_loss', verbose=2, save_best_only=True)] # patience=30


cvae.fit([x_train, to_categorical(y_train)], x_train, epochs=epochs, verbose=2, batch_size=batch_size, shuffle='TRUE',
         validation_split = 0.1, callbacks=my_callbacks)


decoder.save('CVAE_DECODER_model_900_CNN.h5')

print("DONE...")

n_z = 2
n_y = 6

def construct_numvec(digit):
    out = np.zeros((1,  n_y))
    out[:, digit ] = 1.
    return out


decoder = load_model('CVAE_DECODER_model_900_CNN.h5')

train = pd.read_csv('train_data_rp_3_IMBALANCED.csv')
x_train = train.iloc[:, :-1]
x_train = x_train.values
y_train = train.iloc[:, -1:]
y_train = y_train.values
train_imbalanced = np.hstack([x_train, y_train])

n_pixels = 900
n_sample = 6732
mu, sigma = 0.0, 1.0

z1 = np.random.normal(mu, sigma, n_sample)
z2 = np.random.normal(mu, sigma, n_sample)
targets = [int(0), int(4), int(5)]

train_balanced = np.zeros((1, n_pixels+1))
for zone in targets:
    decoded_all = np.zeros((1, n_pixels))
    for i in range(len(z1)):
        vec = construct_numvec(zone)
        z_sample = np.array([[z1[i], z2[i]]])
        decoded = decoder.predict([z_sample, vec])
        decoded = decoded.reshape((1, 900))
        decoded_all = np.vstack([decoded_all, decoded])
    decoded_all = np.delete(decoded_all, (0), axis=0)
    zones = np.ones((n_sample,1))
    zones = zones*zone
    print(zones)
    decoded_all = np.hstack([decoded_all, zones])
    train_balanced = np.vstack([train_balanced, decoded_all])

train_balanced = np.delete(train_balanced, (0), axis=0)

train_CVAE = np.vstack([train_balanced, train_imbalanced])
train_CVAE = shuffle(train_CVAE, random_state=42)
print(train_CVAE.shape)
train_CVAE = pd.DataFrame(train_CVAE)
train_CVAE.to_csv('train_data_rp_3_CVAE.csv', index=False)
Y_cvae = train_CVAE.iloc[:, -1:]
Y_cvae = Y_cvae.values
Y_cvae = Y_cvae.reshape((Y_cvae.shape[0],))
print(sorted(Counter(Y_cvae).items()))
print("DONE...")
