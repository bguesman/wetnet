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

		# 2) Define convolutional layers.
		self.conv_1 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, \
			strides=1, padding='same')
		self.conv_2 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, \
			strides=1, padding='same')
		self.conv_3 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=3, \
			strides=1, padding='same')

		# For now, just do nothing.

	@tf.function
	def call(self, lo_res):
		return self.conv_3(self.conv_2(self.conv_1(lo_res)))

	def loss(self, upsampled, true_hi_res):
		return tf.reduce_sum((upsampled - true_hi_res)**2)
