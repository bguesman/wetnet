import numpy as np
import tensorflow as tf

class FluidAutoencoder(tf.keras.Model):
    def __init__(self, hi_res_dim):

    ######vvv DO NOT CHANGE vvvv##############
        super(FluidAutoencoder, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters

        # Define batch size and optimizer/learning rate
        self.batch_size = 5
        
        self.hi_res_dim = hi_res_dim

        self.rnn_size = 100
        self.RNN = tf.keras.layers.GRU(self.rnn_size,return_sequences=True,return_state=True, recurrent_initializer='glorot_uniform')
        self.linear = tf.keras.layers.Dense(8)
        #2) Define convolutional layers + batch norms.
        self.conv_1 = tf.keras.layers.Conv2D(filters=4, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.conv_2 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(axis=3, scale=True)

        self.deconv_1 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.deconv_2 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.batch_norm_4 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        
        # Global scale to put velocities in right range.
        self.global_scale = tf.Variable(1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    @tf.function
    def call(self, lo_res):

        convs = self.batch_norm_2(self.conv_2( \
            self.batch_norm_1( \
            self.conv_1(lo_res) \
            ) \
            ))
        # frames, width, height, channels
        frames = tf.reshape(convs, (lo_res.shape[0], -1, 8))
        # width * height, frames, channels
        frames = tf.transpose(frames, (1, 0, 2))
        frames = tf.reshape(frames, (frames.shape[0] // 4, -1, 8))
        frames, _ = self.RNN(frames)
        frames = self.linear(frames)

        frames = tf.transpose(frames, (1, 0, 2))
        frames = tf.reshape(frames, (lo_res.shape[0], 38, 38, 8))

        deconv = self.global_scale * self.batch_norm_4( \
            self.deconv_2( \
            self.batch_norm_3 ( \
            self.deconv_1( frames ) ) ) )

        return deconv
        # return np.zeros(lo_res.shape, dtype=np.float32)

    
    def old_loss(self, upsampled, true_hi_res):
        return tf.reduce_sum((upsampled - true_hi_res)**2)
        
    def loss(self, upsampled, true_hi_res, logits_fake):
        return tf.reduce_mean((upsampled - true_hi_res)**2) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_fake), logits=logits_fake))
