import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from fluid_autoencoder import FluidAutoencoder
import sys

import time


def train(model, train_low, train_hi):
	"""
	Runs through one epoch - all training examples.

	:param model: the initilized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, 14)
	:param train_english: english train data (all data for training) of shape (num_sentences, 15)
	:param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
	:return: None
	"""

	# NOTE: For each training step, you should pass in the french sentences to be used by the encoder,
	# and english sentences to be used by the decoder
	# - The english sentences passed to the decoder have the last token in the window removed:
	#	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP]
	#
	# - When computing loss, the decoder labels should have the first word removed:
	#	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

    # For each batch.
	num_batches = int(train_low.shape[0] / model.batch_size)
	print("Training on", num_batches, "batches.")
	
	for i in range(train_low.shape[0]):
		# Collect batch.
		if(model.batch_size + i +1> train_low.shape[0]):
			break
		inputs = train_low[i:model.batch_size + i, :]
		labels = train_hi[i+1:model.batch_size + i+1, :]
		with tf.GradientTape() as tape:
			upsampled = model(inputs)
			loss = model.loss(upsampled, labels)

		# Optimize.
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		if (i % 1 == 0):
            # Pick 100 random datapoints.
			random_data = np.random.randint(0, high=train_low.shape[0],
				size=model.batch_size*10)
			test_loss = test(model, tf.gather(train_low, random_data),
				tf.gather(train_hi, random_data))
			print("Batch", i, ", average loss on random", model.batch_size*10,
				"datapoints: ", test_loss)

def test(model, test_low, test_hi):
	"""
	Runs through one epoch - all testing examples.
	"""
	avg_loss = 0
	num_batches = int(test_low.shape[0] / model.batch_size)
	for i in range(test_low.shape[0]):
		# Collect batch.
		if(model.batch_size + i +1> test_low.shape[0]):
			break
		batch_inputs = test_low[i:model.batch_size + i, :]
		batch_labels = test_hi[i+1:model.batch_size + i+1, :]
		# Compute loss.
		upsampled = model(batch_inputs)
		print("Low max:", np.max(batch_inputs))
		print("Upsampled max:", np.max(batch_labels))
		loss = model.loss(upsampled, batch_labels)
		# Accumulate loss.
		avg_loss += loss / model.batch_size

	return avg_loss / num_batches

def main():

	print("Running preprocessing...")
	train_low, train_hi, test_low, test_hi = get_data('../data/lo_res/', '../data/hi_res/')
	print("Preprocessing complete.")

	print("Lo-res dimension:", test_low.shape[:])
	print("Hi-res dimension:", test_hi.shape[:])

	model = FluidAutoencoder(test_hi.shape[:])

	# Train and Test Model.
	start = time.time()
	epochs = 1
	for i in range(epochs):
		train(model, train_low, train_hi)
	end = time.time()
	print("Done training, took", (end - start) / 60, "minutes.")

	loss = test(model, test_low, test_hi)
	print("FINAL LOSS ON TEST DATA:", loss)

	model.save_weights('model_weights/model_weights', save_format='tf')

if __name__ == '__main__':
   main()
