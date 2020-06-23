import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = top_k + 1
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

# Load the numpy files
def map_func(img_name, cap):
	#load .npy extracted from inceptionV3
	img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	return img_tensor, cap

def createDataset():
	dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

	# Use map to load the numpy files in parallel
	dataset = dataset.map(lambda item1, item2: tf.numpy_function(
			map_func, [item1, item2], [tf.float32, tf.int32]),
			num_parallel_calls=tf.data.experimental.AUTOTUNE)

	# Shuffle and batch
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)