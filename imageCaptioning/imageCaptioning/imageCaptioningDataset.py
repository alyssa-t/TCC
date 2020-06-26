import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape

class Methods_Dataset:
# Load the numpy files
	def __init__ (self, img_name_vector, cap_vector, batch_size=64, buffer_size=1000):
		self.img_name_vector = img_name_vector
		self.cap_vector = cap_vector
		self.batch_size= batch_size
		self.buffer_size = buffer_size

	def map_func(self, img_name, cap):
		#load .npy extracted from inceptionV3
		img_tensor = np.load(img_name.decode('utf-8')+'.npy')
		return img_tensor, cap

	def createDataset(self):
		dataset = tf.data.Dataset.from_tensor_slices((self.img_name_vector, self.cap_vector))
		# Use map to load the numpy files in parallel
		dataset = dataset.map(lambda item1, item2: tf.numpy_function(
								self.map_func, [item1, item2], [tf.float32, tf.int32]),
								num_parallel_calls=tf.data.experimental.AUTOTUNE)
		# Shuffle and batch
		dataset = dataset.shuffle(self.buffer_size).batch(self.batch_size)
		dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
		return dataset