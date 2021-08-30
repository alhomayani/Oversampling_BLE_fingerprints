# code inspired from: https://keras.io/examples/generative/vae/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import os
import pandas as pd
from collections import Counter
from imblearn.datasets import make_imbalance
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.utils import shuffle


class Sampling(layers.Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 2

encoder_inputs = keras.Input(shape=(30, 30, 1))
x = layers.Conv2D(8, 4, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(16, 4, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(8, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(8 * 8 * 16, activation="relu")(latent_inputs)
x = layers.Reshape((8, 8, 16))(x)
x = layers.Conv2DTranspose(16, 4, activation="relu", strides=2, padding="same", output_padding=1)(x)
x = layers.Conv2DTranspose(8, 4, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        z_mean, z_log_var, z = encoder(data)
        reconstruction = decoder(z)
        reconstruction_loss = tf.reduce_mean(
            keras.losses.binary_crossentropy(data, reconstruction)
        )
        reconstruction_loss *= 30 * 30
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        total_loss = reconstruction_loss + kl_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


minorities = [int(0), int(4), int(5)]

for minority in minorities:
    os.chdir('/content/gdrive/My Drive/training_testing_data/')
    train = pd.read_csv('train_data_rp_3_IMBALANCED.csv')
    train.columns = [*train.columns[:-1], 'zone']
    train = train[train['zone'] == minority]
    train = train.sample(frac=1).reset_index(drop=True)
    x_train = train.iloc[:, :-1]
    x_train = x_train.values
    Y_train = train.iloc[:, -1:]
    Y_train = Y_train.values
    x_train = x_train.reshape((x_train.shape[0], 30, 30, 1))

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    my_callbacks = [EarlyStopping(patience=3000), ModelCheckpoint('model_vae.h5', monitor='val_loss', verbose=2,
                                                                  save_best_only=True)]

    vae.fit(x_train, epochs=30000, batch_size=23, verbose=2, callbacks=my_callbacks, shuffle='TRUE',
            validation_split=0.1)
    decoder.save('VAE_DECODER_model_900_CNN_' + str(minority) + '.h5')

    print(str(minority) + "DONE...")

os.chdir('/content/gdrive/My Drive/training_testing_data/')

n_pixels = 900
train = pd.read_csv('train_data_rp_3_IMBALANCED.csv')
x_train = train.iloc[:, :-1]
x_train = x_train.values
y_train = train.iloc[:, -1:]
y_train = y_train.values
train_imbalanced = np.hstack([x_train, y_train])


def sample_latent_space(minority):
    print("working on minority: ", minority)
    n_sample = 6732
    mu, sigma = 0.0, 1.0

    z1 = np.random.normal(mu, sigma, n_sample)
    z2 = np.random.normal(mu, sigma, n_sample)

    decoder = load_model('VAE_DECODER_model_900_CNN_' + str(minority) + '.h5')

    decoded_all = np.zeros((1, n_pixels))

    for i in range(len(z1)):
        z_sample = np.array([[z1[i], z2[i]]])
        x_decoded = decoder.predict(z_sample)
        decoded = x_decoded.reshape((1, n_pixels))
        decoded_all = np.vstack([decoded_all, decoded])

    decoded_all = np.delete(decoded_all, (0), axis=0)
    zones = np.ones((6732, 1))
    zones = zones * minority
    decoded_all = np.hstack([decoded_all, zones])

    decoded_all = pd.DataFrame(decoded_all)
    decoded_all.to_csv('train_data_rp_3_VAE_' + str(minority) + '.csv', index=False)
    Y_vae = decoded_all.iloc[:, -1:]
    Y_vae = Y_vae.values
    Y_vae = Y_vae.reshape((Y_vae.shape[0],))
    print(sorted(Counter(Y_vae).items()))


for minority in minorities:
    sample_latent_space(minority)

npa_all = np.zeros((1, n_pixels + 1))
for minority in minorities:
    df = pd.read_csv('train_data_rp_3_VAE_' + str(minority) + '.csv')
    npa = df.values
    npa_all = np.vstack([npa_all, npa])

npa_all = np.delete(npa_all, (0), axis=0)
npa_all = np.vstack([npa_all, train_imbalanced])
npa_all = shuffle(npa_all)

print(npa_all.shape)
train_VAE = pd.DataFrame(npa_all)
train_VAE.to_csv('train_data_rp_3_VAE.csv', index=False)
Y_vae = train_VAE.iloc[:, -1:]
Y_vae = Y_vae.values
Y_vae = Y_vae.reshape((Y_vae.shape[0],))
print("ALL:", sorted(Counter(Y_vae).items()))
print("ALL: DONE...")


