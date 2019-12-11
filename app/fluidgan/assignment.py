import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from fluid_autoencoder import FluidAutoencoder
import sys

import time


def train(model, train_low, train_hi, train_d):
	"""
	Runs through one epoch - all training examples.
	"""

	print("Training", train_low.shape[0]-1, "batches.")
	
	for k in range(20):
		i = np.random.randint(1, high=train_low.shape[0]-1-model.batch_size, size=1)[0]
		# Collect batch.
		if(model.batch_size + i + 1 > train_low.shape[0]):
			break
		inputs = train_low[i+1:model.batch_size + i + 1, :]
		density_tn1 = train_d[i:model.batch_size + i, :]
		labels_tn1 = train_hi[i:model.batch_size + i, :]
		labels_t0 = train_hi[i+1:model.batch_size + i+1, :]
		labels_t1 = train_hi[i+2:model.batch_size + i + 2, :]
		with tf.GradientTape() as tape:
			# print("model")
			upsampled = inputs + model(inputs)
			# print("loss")
			loss = model.loss(upsampled, labels_tn1, labels_t0, labels_t1, density_tn1)

		# Optimize.
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		if (k == 19):
			model.save_weights('model_weights/model_weights', save_format='tf')

		if (i % 1 == 0):
            # Pick random contiguous datapoints.
			datapoints = model.batch_size*10
			random_index = np.random.randint(1, high=train_low.shape[0]-1-datapoints, size=1)
			random_data = np.arange(random_index, random_index + model.batch_size*10)
			test_loss = test(model, tf.gather(train_low, random_data),
				tf.gather(train_hi, random_data-1), 
				tf.gather(train_hi, random_data), 
				tf.gather(train_hi, random_data+1),
				tf.gather(train_d, random_data-1))
			print("Batch", k, ", average loss on random", model.batch_size*10,
				"datapoints: ", test_loss)
			print("Index of loss:", random_index)

def test(model, test_low, test_hi_tn1, test_hi_t0, test_hi_t1, test_d):
	"""
	Runs through one epoch - all testing examples.
	"""
	avg_loss = 0
	num_batches = int(test_low.shape[0] / model.batch_size)
	for i in range(num_batches):
		# Collect batch.
		if (model.batch_size + i +1> test_low.shape[0]):
			break
		batch_inputs = test_low[i:model.batch_size + i, :]
		batch_d = test_d[i:model.batch_size + i, :]
		batch_labels_tn1 = test_hi_tn1[i:model.batch_size + i, :]
		batch_labels_t0 = test_hi_t0[i:model.batch_size + i, :]
		batch_labels_t1 = test_hi_t1[i:model.batch_size + i, :]
		# Compute loss.
		upsampled = batch_inputs + model(batch_inputs)
		# print("Low max:", np.max(batch_inputs))
		# print("Upsampled max:", np.max(batch_labels_t0))
		loss = model.loss(upsampled, batch_labels_tn1, batch_labels_t0, batch_labels_t1, batch_d)
		# Accumulate loss.
		avg_loss += loss / model.batch_size

	return avg_loss / num_batches

def main():

	model = FluidAutoencoder()

	# Train and Test Model.
	start = time.time()
	epochs = 10
	offset = 600
	frame_block_size = 400
	frame_blocks = 1200 // frame_block_size
	for i in range(epochs):
		for j in range(frame_blocks):
			print("Loading frame block", j, "...")
			train_low, train_hi, train_d, test_low, test_hi, test_d = \
				get_data('../data/lo_res/', '../data/hi_res/', offset + j * frame_block_size, \
				frame_block_size)
			print("Frame block loaded.")
			print("Lo-res dimension:", test_low.shape[:])
			print("Hi-res dimension:", test_hi.shape[:])
			train(model, train_low, train_hi, train_d)
	end = time.time()
	print("Done training, took", (end - start) / 60, "minutes.")

	loss = test(model, test_low, test_hi)
	print("FINAL LOSS ON TEST DATA:", loss)

	model.save_weights('model_weights/model_weights', save_format='tf')

if __name__ == '__main__':
   main()
