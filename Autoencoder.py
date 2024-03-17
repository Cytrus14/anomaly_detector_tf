import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

class Autoencoder(Model):
    def __init__(self, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.encoder = tf.keras.Sequential([
            layers.Dense(36, activation="relu"),
            layers.Dense(14, activation="relu")
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(36, activation='relu'),
            layers.Dense(52, activation="sigmoid")
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# autoencoder = Autoencoder()
# autoencoder.compile(optimizer="adam", loss="mae")
# history = autoencoder.fit()