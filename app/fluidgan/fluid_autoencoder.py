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
		self.batch_size = 10
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		self.hi_res_dim = hi_res_dim
		self.regularizer = 1.0

		# 2) Define convolutional layers.
		self.conv_1 = tf.keras.layers.Conv2D(filters=4, kernel_size=5, \
			strides=2, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
		self.conv_2 = tf.keras.layers.Conv2D(filters=8, kernel_size=5, \
			strides=2, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
		self.deconv_1 = tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=5, \
			strides=2, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))
		self.deconv_2 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=5, \
			strides=2, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001))

	@tf.function
	def call(self, lo_res):
		return self.deconv_2(self.deconv_1(self.conv_2(self.conv_1(lo_res))))

	def loss(self, low_res, true_hi_res, velocity_diffs):
		return tf.reduce_sum(((low_res + velocity_diffs) - true_hi_res)**2) \
			+ self.regularizer * tf.reduce_sum(velocity_diffs ** 2)
