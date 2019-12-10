import numpy as np
import tensorflow as tf
import numpy as np
import cv2

def get_data(lo_res_file, hi_res_file):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.
	"""

	frame = 0
	low_v = []
	hi_v = []
	while frame < 2000:
		try:
			new_low_v = np.load(lo_res_file + format(frame, "0>9") + ".npz")['v']
			new_hi_v = np.load(hi_res_file + format(frame, "0>9") + ".npz")['v']
			low_v.append(np.array(cv2.resize(new_low_v, dsize=(160, 160),
	            interpolation=cv2.INTER_LINEAR), dtype=np.float32))
			hi_v.append(np.array(cv2.resize(new_hi_v, dsize=(160, 160),
	            interpolation=cv2.INTER_LINEAR), dtype=np.float32))
		except:
			print("exception!!!")
			break
		frame += 1

	low_v = np.stack(low_v)
	hi_v = np.stack(hi_v)

	print("number of low datapoints:", low_v.shape[0])
	print("number of hi datapoints:", hi_v.shape[0])

	assert low_v.shape[0] == hi_v.shape[0]


	# Split into train and test. 90% -> train, 10% -> test.
	lo_res_train = low_v[:int(-low_v.shape[0]/10)]
	lo_res_test= low_v[int(-low_v.shape[0]/10):]
	hi_res_train = hi_v[:int(-hi_v.shape[0]/10)]
	hi_res_test= hi_v[int(-hi_v.shape[0]/10):]

	return lo_res_train, hi_res_train, lo_res_test, hi_res_test
