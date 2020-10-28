import numpy as np

import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()
import tensorflow.keras
from tensorflow.keras.layers import Input,Conv2D,Flatten,Dense,Lambda,Reshape,Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
        

class VAE(tensorflow.keras.Model):

    def __init__(self,img_shape, latent_dim, beta):
        super(VAE, self).__init__(name='vae')
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.z_m = None
        self.z_s = None

        self.encoder, self.decoder, self.vae  = self.build_model()


    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)

        return z_mean + self.beta * K.exp(z_log_var) * epsilon

    
    def build_model(self):
        # Encoder
        input_img = Input(shape=self.img_shape)
        x = Conv2D(32, 3, padding='same', activation='relu')(input_img)
        x = Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        shape_before_flattening = K.int_shape(x)
        x = Flatten()(x)
        x = Dense(32, activation='relu')(x)

        z_mean = Dense(self.latent_dim)(x) # outputの次元数：潜在変数の数
        z_log_var = Dense(self.latent_dim)(x)

        self.z_m = z_mean  # for Loss
        self.z_s = z_log_var # for Loss

        z = Lambda(self.sampling)([z_mean, z_log_var])
        
        encoder = Model(input_img, z_mean)

        # Decoder
        decoder_input = Input(K.int_shape(z)[1:])
        x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(shape_before_flattening[1:])(x)
        x = Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
        x = Conv2D(3, 3, padding='same', activation='sigmoid')(x)

        decoder = Model(decoder_input, x)
        z_decoded = decoder(z)
    
        vae = Model(input_img, z_decoded)

        # y = CustomVariationalLayer()([input_img, z_decoded, z_mean, z_log_var])
        # vae = Model(input_img, y)

        return encoder,decoder,vae


    def binary_crossentropy(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)


    def vae_loss(self, x, x_decoded_mean):
        z_mean = self.z_m
        z_log_var = self.z_s

        latent_loss =  - 5e-4 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
        reconst_loss = K.mean(self.binary_crossentropy(x, x_decoded_mean),axis=-1)

        # z_sigma = self.z_s
        # latent_loss =  - 0.5 * K.mean(K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1))
        
        return latent_loss + reconst_loss

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_vae(self):
        self.vae.compile(optimizer='adam', loss=self.vae_loss, experimental_run_tf_function=False)
        return self.vae
        