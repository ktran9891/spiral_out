import numpy as np
import h5py
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping


# Read the features
h5f = h5py.File('features.h5', 'r')
features = h5f['ResNet_features'][:]
features = features[:, 0, :]
h5f.close()

# Split the data up
train_size = 0.8
features_train, features_validate = train_test_split(features,
                                                     train_size=train_size,
                                                     test_size=1-train_size,
                                                     random_state=42)

def train_encoder(x_train, x_validate,
                  n_latent_vars, latent_activation, decoder_activation,
                  optimizer, loss, epochs, batch_size):
    # Make the tensors so that we can define the *coders
    n_dimen = x_train.shape[1]
    input_shape = (n_dimen,)
    input_tensor = Input(shape=input_shape)
    latent_tensor = Dense(n_latent_vars, activation=latent_activation)(input_tensor)
    output_tensor = Dense(n_dimen, activation=decoder_activation)(latent_tensor)

    # Make and train the autoencoder
    encoder = Model(input_tensor, latent_tensor)
    autoencoder = Model(input_tensor, output_tensor)
    autoencoder.compile(optimizer=optimizer, loss=loss)
    stopper = EarlyStopping(patience=3)
    autoencoder.fit(x_train, x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(x_validate, x_validate),
                    callbacks=[stopper])
    return autoencoder, encoder

# Train shallow [auto]encoder
n_latent_vars = 16
autoencoder, encoder = train_encoder(x_train=features_train,
                                     x_validate=features_validate,
                                     n_latent_vars=n_latent_vars,
                                     latent_activation='softplus',
                                     decoder_activation='softplus',
                                     optimizer='adam',
                                     loss='mean_squared_error',
                                     epochs=100,
                                     batch_size=16)

# Save the autoencoder and encoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.save('autoencoder.h5')
encoder.compile(optimizer='adam', loss='mean_squared_error')
encoder.save('encoder.h5')

# And then save the encoded features
encoded_features = encoder.predict(features)
h5f = h5py.File('features.h5', 'a')
try:
    h5f.create_dataset('encoded_features', data=encoded_features)
except RuntimeError:
    h5_data = h5f['encoded_features']
    h5_data = features
h5f.close()
