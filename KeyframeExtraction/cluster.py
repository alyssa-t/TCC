import numpy as np
import pandas as pd
import cv2
import os
import shutil
import time
import gc
from tqdm import tqdm
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from progressbar import ProgressBar


class Image_Clustering:
	def __init__(self, OPTIC_MIN_SAMPLES):
		super(Image_Clustering, self).__init__()
		self.minSamples = OPTIC_MIN_SAMPLES

	def clust(self, arrayFrames):
		print('Label images...')
		# Load a model
		model = VGG16(weights='imagenet', include_top=False)
		X = self.__feature_extraction(arrayFrames, model, 64)
		# Clutering images by OPTICS
		op = OPTICS(min_samples=self.minSamples).fit(X)
		selectedLabels = [-1] # -1 = noise cluster.
		imageIndexes = []
		for i, label in enumerate(op.labels_, start=0):
			if label not in selectedLabels:
				selectedLabels.append(label)
				imageIndexes.append(i)
		return imageIndexes

	def __feature_extraction(self, arrayFrames, model, batchSize):
		batch_features = model.predict(arrayFrames)
		batch_features = tf.reshape(batch_features, (arrayFrames.shape[0], -1))
		return batch_features