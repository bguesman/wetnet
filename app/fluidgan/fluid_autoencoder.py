import numpy as np
import tensorflow as tf
from scipy import ndimage

class FluidAutoencoder(tf.keras.Model):
	def __init__(self):

    ######vvv DO NOT CHANGE vvvv##############
		super(FluidAutoencoder, self).__init__()
		######^^^ DO NOT CHANGE ^^^##################


		# TODO:
		# 1) Define any hyperparameters

		# Define batch size and optimizer/learning rate
		self.batch_size = 8
		
		self.dt = 2 # HACK! In smokemultires.py

		self.rnn_size = 10 * 10
		self.RNN0 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN1 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN2 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN3 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN4 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN5 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN6 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN7 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN8 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN9 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN10 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN11 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN12 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN13 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN14 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.RNN15 = tf.keras.layers.GRU(self.rnn_size, return_sequences=True,return_state=False, 
				recurrent_initializer='glorot_uniform', stateful=False, unroll=True)
		self.linear = tf.keras.layers.Dense(16)
		
		#2) Define convolutional layers + batch norms.
		self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=7, \
			strides=4, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
			activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		self.batch_norm_1 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
		self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=7, \
			strides=4, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
			activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		self.batch_norm_2 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
		# self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, \
		# 	strides=2, padding='same',
		# 	kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
		# 	activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		# self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
		# self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=7, \
		# 	strides=2, padding='same',
		# 	kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
		# 	activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		# self.batch_norm_4 = tf.keras.layers.BatchNormalization(axis=3, scale=True)


		self.deconv1 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=7, \
			strides=4, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
			activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		self.batch_norm_3 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
		self.deconv2 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=7, \
			strides=4, padding='same',
			kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
			activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		# self.batch_norm_6 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
		# self.deconv3 = tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=7, \
		# 	strides=2, padding='same',
		# 	kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
		# 	activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		# self.batch_norm_7 = tf.keras.layers.BatchNormalization(axis=3, scale=True)
		# self.deconv4 = tf.keras.layers.Conv2DTranspose(filters=2, kernel_size=7, \
		# 	strides=2, padding='same',
		# 	kernel_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
		# 	activation=tf.keras.layers.LeakyReLU(alpha=0.2))
		
		# Global scale to put velocities in right range.
		self.global_scale = tf.Variable(1.0)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)

	@tf.function
	def call(self, lo_res):
		# print("call")
		# convs = self.batch_norm_4(self.conv4(self.batch_norm_3(self.conv3(self.batch_norm_2(self.conv2(self.batch_norm_1(self.conv1(lo_res))))))))
		convs = self.batch_norm_2(self.conv2(self.batch_norm_1(self.conv1(lo_res))))
		#frames = tf.image.extract_patches(convs, [1, convs.shape(1) // 2, convs.shape(2) // 2, 1], 
		#	[1, convs.shape(1) // 2, convs.shape(2) // 2, 1], [1, 1, 1, 1], "same")
		# frames, width * height, channels
		frames = tf.reshape(convs, (convs.shape[0], -1, convs.shape[3]))
		frames = tf.transpose(frames, (2, 0, 1))
		# rnn_results = tf.stack([self.RNNs[i](tf.expand_dims(frames[i,:,:], axis=0)) for i in range(16)])
		# rnn_results = tf.zeros([16, 8, 100])
		# print("here")
		# rnn_results0 = tf.squeeze(self.RNN0(tf.expand_dims(frames[0,:,:], axis=0)))
		
		# rnn_results1 = tf.squeeze(self.RNN1(tf.expand_dims(frames[1,:,:], axis=0)))
		# rnn_results2 = tf.squeeze(self.RNN2(tf.expand_dims(frames[2,:,:], axis=0)))
		# rnn_results3 = tf.squeeze(self.RNN3(tf.expand_dims(frames[3,:,:], axis=0)))
		# rnn_results4 = tf.squeeze(self.RNN4(tf.expand_dims(frames[4,:,:], axis=0)))
		# rnn_results5 = tf.squeeze(self.RNN5(tf.expand_dims(frames[5,:,:], axis=0)))
		# rnn_results6 = tf.squeeze(self.RNN6(tf.expand_dims(frames[6,:,:], axis=0)))
		# rnn_results7 = tf.squeeze(self.RNN7(tf.expand_dims(frames[7,:,:], axis=0)))
		# rnn_results8 = tf.squeeze(self.RNN8(tf.expand_dims(frames[8,:,:], axis=0)))
		# rnn_results9 = tf.squeeze(self.RNN9(tf.expand_dims(frames[9,:,:], axis=0)))
		# rnn_results10 = tf.squeeze(self.RNN10(tf.expand_dims(frames[10,:,:], axis=0)))
		# rnn_results11 = tf.squeeze(self.RNN11(tf.expand_dims(frames[11,:,:], axis=0)))
		# rnn_results12 = tf.squeeze(self.RNN12(tf.expand_dims(frames[12,:,:], axis=0)))
		# rnn_results13 = tf.squeeze(self.RNN13(tf.expand_dims(frames[13,:,:], axis=0)))
		# rnn_results14 = tf.squeeze(self.RNN14(tf.expand_dims(frames[14,:,:], axis=0)))
		# rnn_results15 = tf.squeeze(self.RNN15(tf.expand_dims(frames[15,:,:], axis=0)))

		rnn_results0 = tf.squeeze(self.RNN0(tf.expand_dims(frames[0,:,:], axis=0)))
		rnn_results1 = tf.squeeze(self.RNN1(tf.expand_dims(frames[1,:,:], axis=0)))
		rnn_results2 = tf.squeeze(self.RNN2(tf.expand_dims(frames[2,:,:], axis=0)))
		rnn_results3 = tf.squeeze(self.RNN3(tf.expand_dims(frames[3,:,:], axis=0)))
		rnn_results4 = tf.squeeze(self.RNN4(tf.expand_dims(frames[4,:,:], axis=0)))
		rnn_results5 = tf.squeeze(self.RNN5(tf.expand_dims(frames[5,:,:], axis=0)))
		rnn_results6 = tf.squeeze(self.RNN6(tf.expand_dims(frames[6,:,:], axis=0)))
		rnn_results7 = tf.squeeze(self.RNN7(tf.expand_dims(frames[7,:,:], axis=0)))
		rnn_results8 = tf.squeeze(self.RNN8(tf.expand_dims(frames[8,:,:], axis=0)))
		rnn_results9 = tf.squeeze(self.RNN9(tf.expand_dims(frames[9,:,:], axis=0)))
		rnn_results10 = tf.squeeze(self.RNN10(tf.expand_dims(frames[10,:,:], axis=0)))
		rnn_results11 = tf.squeeze(self.RNN11(tf.expand_dims(frames[11,:,:], axis=0)))
		rnn_results12 = tf.squeeze(self.RNN12(tf.expand_dims(frames[12,:,:], axis=0)))
		rnn_results13 = tf.squeeze(self.RNN13(tf.expand_dims(frames[13,:,:], axis=0)))
		rnn_results14 = tf.squeeze(self.RNN14(tf.expand_dims(frames[14,:,:], axis=0)))
		rnn_results15 = tf.squeeze(self.RNN15(tf.expand_dims(frames[15,:,:], axis=0)))

		# print(rnn_results15.shape)

		rnn_results = tf.stack([rnn_results0, rnn_results1,
			rnn_results2, rnn_results3, rnn_results4, rnn_results5, rnn_results6, rnn_results7,
			rnn_results8, rnn_results9, rnn_results10, rnn_results11, rnn_results12, rnn_results13,
			rnn_results14, rnn_results15])


		#rnn_results = tf.transpose(rnn_results, (1, 2, 0)) 
		rnn_results = tf.transpose(rnn_results, (1, 0))# HACK FOR RUNNING
		rnn_results = tf.reshape(rnn_results, convs.shape)
		# print("fuck everything")

		# timesteps, width*height, channels
		# frames = tf.reshape(convs, (lo_res.shape[0], -1, 8))
		# # channels, timesteps, width*height
		# frames = tf.transpose(frames, (2, 0, 1))
		# # channels, timesteps, width * height / 4, 2, 2
		# frames = tf.reshape(frames, (frames.shape[0], frames.shape[1], frames.shape[2]//4, 2, 2))
		# # channels, timesteps, width * height / 4, 2, 2
		# frames, _ = self.RNN(frames)

		# # channels, timesteps, width*height
		# frames = tf.reshape(frames, (frames.shape[0], frames.shape[1], frames.shape[2]))
		# # timesteps, width*height, channels
		# frames = tf.transpose(frames, (1, 2, 0))
		# # timesteps, width, height, channels
		# frames = tf.reshape(frames, (lo_res.shape[0], 38, 38, 8))

		deconv = self.global_scale * self.deconv2(self.batch_norm_1(self.deconv1(rnn_results)))
		# deconv = self.global_scale * self.deconv_2( \
		# 	self.batch_norm_3 ( \
		# 	self.deconv_1( convs ) ) )

		return deconv
		# return np.zeros(lo_res.shape, dtype=np.float32)

	def advect(self, data, v, dim, fill, interp_method, collision=True):
        # Get a grid of cell indices (cell center point locations).
		x_range = np.arange(0, data.shape[0])
		y_range = np.arange(0, data.shape[1])
		xx, yy = np.meshgrid(x_range, y_range)

        # Use x, y to fit with velocity grid's order.
		grid = np.stack([np.transpose(xx), np.transpose(yy)], axis=-1)

        # Trace those points backward in time using the velocity field.
		backtraced_locations = grid - self.dt * v 
		if (collision):
			backtraced_locations = np.abs(backtraced_locations)

        # Sample the velocity at those points, set it to the new velocity.
		backtraced_locations_reshaped = backtraced_locations.reshape(-1,2).transpose()
		if (dim == 2):
			interpolated_x = ndimage.map_coordinates(data[:,:,0],
				backtraced_locations_reshaped, order=1, mode='constant', cval=fill)
			interpolated_y = ndimage.map_coordinates(data[:,:,1],
				backtraced_locations_reshaped, order=1, mode='constant', cval=fill)
			interpolated = np.stack([interpolated_x, interpolated_y], axis=-1)
		else:
			interpolated = ndimage.map_coordinates(data,
				backtraced_locations_reshaped, order=1, mode='constant', cval=fill)

        # Make sure to reshape back to a grid!
		interpolated = interpolated.reshape(data.shape)

		return interpolated

	def loss(self, upsampled, hi_res_tn1, hi_res_t0, hi_res_t1, hi_res_d):
		advected_backward = np.zeros(upsampled.shape, dtype=np.float32)
		advected_backward[0,:,:,:] = self.advect(upsampled[0,:,:,:], -upsampled[0,:,:,:], 2, 0.0, 'linear')
		advected_backward[1,:,:,:] = self.advect(upsampled[1,:,:,:], -upsampled[1,:,:,:], 2, 0.0, 'linear')
		advected_backward[2,:,:,:] = self.advect(upsampled[2,:,:,:], -upsampled[2,:,:,:], 2, 0.0, 'linear')
		advected_backward[3,:,:,:] = self.advect(upsampled[3,:,:,:], -upsampled[3,:,:,:], 2, 0.0, 'linear')
		advected_backward[4,:,:,:] = self.advect(upsampled[4,:,:,:], -upsampled[4,:,:,:], 2, 0.0, 'linear')
		advected_backward[5,:,:,:] = self.advect(upsampled[5,:,:,:], -upsampled[5,:,:,:], 2, 0.0, 'linear')
		advected_backward[6,:,:,:] = self.advect(upsampled[6,:,:,:], -upsampled[6,:,:,:], 2, 0.0, 'linear')
		advected_backward[7,:,:,:] = self.advect(upsampled[7,:,:,:], -upsampled[7,:,:,:], 2, 0.0, 'linear')

		advected_forward = np.zeros(upsampled.shape, dtype=np.float32)
		advected_forward[0,:,:,:] = self.advect(upsampled[0,:,:,:], upsampled[0,:,:,:], 2, 0.0, 'linear')
		advected_forward[1,:,:,:] = self.advect(upsampled[1,:,:,:], upsampled[1,:,:,:], 2, 0.0, 'linear')
		advected_forward[2,:,:,:] = self.advect(upsampled[2,:,:,:], upsampled[2,:,:,:], 2, 0.0, 'linear')
		advected_forward[3,:,:,:] = self.advect(upsampled[3,:,:,:], upsampled[3,:,:,:], 2, 0.0, 'linear')
		advected_forward[4,:,:,:] = self.advect(upsampled[4,:,:,:], upsampled[4,:,:,:], 2, 0.0, 'linear')
		advected_forward[5,:,:,:] = self.advect(upsampled[5,:,:,:], upsampled[5,:,:,:], 2, 0.0, 'linear')
		advected_forward[6,:,:,:] = self.advect(upsampled[6,:,:,:], upsampled[6,:,:,:], 2, 0.0, 'linear')
		advected_forward[7,:,:,:] = self.advect(upsampled[7,:,:,:], upsampled[7,:,:,:], 2, 0.0, 'linear')
		
		# Advect forward and backward.
		# advected_backward = [self.advect(upsampled[i,:,:,:], -upsampled[i,:,:,:], 2, 0.0, 'linear') for i in range(upsampled.shape[0])]
		# advected_forward = [self.advect(upsampled[i,:,:,:], upsampled[i,:,:,:], 2, 0.0, 'linear') for i in range(upsampled.shape[0])]
		
		advected_backward = tf.convert_to_tensor(advected_backward)
		advected_forward = tf.convert_to_tensor(advected_forward)

		# Advect density according to true high res and upsampled velocity fields.
		advected_hi_res_density = np.zeros(hi_res_d.shape, dtype=np.float32)
		advected_hi_res_density[0,:,:,:] = self.advect(hi_res_d[0,:,:], hi_res_t0[0,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[1,:,:,:] = self.advect(hi_res_d[1,:,:], hi_res_t0[1,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[2,:,:,:] = self.advect(hi_res_d[2,:,:], hi_res_t0[2,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[3,:,:,:] = self.advect(hi_res_d[3,:,:], hi_res_t0[3,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[4,:,:,:] = self.advect(hi_res_d[4,:,:], hi_res_t0[4,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[5,:,:,:] = self.advect(hi_res_d[5,:,:], hi_res_t0[5,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[6,:,:,:] = self.advect(hi_res_d[6,:,:], hi_res_t0[6,:,:,:], 1, 0.0, 'linear')
		advected_hi_res_density[7,:,:,:] = self.advect(hi_res_d[7,:,:], hi_res_t0[7,:,:,:], 1, 0.0, 'linear')
		
		advected_upsampled_density = np.zeros(hi_res_d.shape, dtype=np.float32)
		advected_upsampled_density[0,:,:,:] = self.advect(hi_res_d[0,:,:], upsampled[0,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[1,:,:,:] = self.advect(hi_res_d[1,:,:], upsampled[1,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[2,:,:,:] = self.advect(hi_res_d[2,:,:], upsampled[2,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[3,:,:,:] = self.advect(hi_res_d[3,:,:], upsampled[3,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[4,:,:,:] = self.advect(hi_res_d[4,:,:], upsampled[4,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[5,:,:,:] = self.advect(hi_res_d[5,:,:], upsampled[5,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[6,:,:,:] = self.advect(hi_res_d[6,:,:], upsampled[6,:,:,:], 1, 0.0, 'linear')
		advected_upsampled_density[7,:,:,:] = self.advect(hi_res_d[7,:,:], upsampled[7,:,:,:], 1, 0.0, 'linear')
		
		density_loss = tf.reduce_sum((advected_upsampled_density - advected_hi_res_density)**2)
		forward_temporal_loss = tf.reduce_sum((advected_forward - hi_res_t1) ** 2)
		backward_temporal_loss = tf.reduce_sum((advected_backward - hi_res_tn1) ** 2)
		spatial_loss = tf.reduce_sum((upsampled - hi_res_t0)**2)
		return forward_temporal_loss + backward_temporal_loss + spatial_loss + density_loss
