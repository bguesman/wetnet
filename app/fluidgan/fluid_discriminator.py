import numpy as np
import tensorflow as tf

class FluidDiscriminator(tf.keras.Model):
    def __init__(self, hi_res_dim):

    ######vvv DO NOT CHANGE vvvv##############
        super(FluidDiscriminator, self).__init__()
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters

        # Define batch size and optimizer/learning rate
        self.batch_size = 5
        
        self.hi_res_dim = hi_res_dim

        
        # Discrim layers
        self.discrim_conv_1 = tf.keras.layers.Conv2D(filters=4, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.discrim_batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.discrim_conv_2 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, \
            strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.discrim_batch_norm_2 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
        self.discrim_final = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001), 
            bias_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

    @tf.function
    def call(self, data):
        convs = self.discrim_batch_norm_2(self.discrim_conv_2( \
            self.discrim_batch_norm_1( \
            self.discrim_conv_1(data) \
            ) \
            ))

        return self.discrim_final(convs)

    
    def loss(self, logits_real, logits_fake):
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(logits_fake), logits=logits_fake))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits_real), logits=logits_real))
        return D_loss
