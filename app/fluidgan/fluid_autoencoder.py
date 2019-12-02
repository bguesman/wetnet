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
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
		self.hi_res_dim = hi_res_dim

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


	@tf.function
	def call(self, lo_res):
		return self.global_scale * self.batch_norm_4( \
			self.deconv_2( \
			self.batch_norm_3 ( \
			self.deconv_1( \
			self.batch_norm_2 ( \
			self.conv_2( \
			self.batch_norm_1( \
			self.conv_1(lo_res) \
			) \
			) \
			) \
			) \
			) \
			) \
			) \
		# return np.zeros(lo_res.shape, dtype=np.float32)

	def loss(self, upsampled, true_hi_res):
		return tf.reduce_sum((upsampled - true_hi_res)**2)
