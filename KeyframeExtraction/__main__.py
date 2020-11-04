import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
import pickle
import warnings
import glob

from PIL import Image
from tqdm import tqdm

from .extractFrames import *
from .cluster import *

VIDEO_FILE_NAME = '/.../uptown.mp4'                     # Video file name
TARGET_IMAGES_DIR = '/.../'     						# save keyframes
FRAME_PER_SECONDS = 1
RESIZED_IMAGE_SIZE = 224
OPTIC_MIN_SAMPLES = 2


def main():
	#Code for setup GPU
	#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	gpu_devices = tf.config.experimental.list_physical_devices('GPU')
	for device in gpu_devices:
		tf.config.experimental.set_memory_growth(device, True)
	tf.compat.v1.enable_eager_execution()
	
	#start_time = time.time()
	extractKF = Data2Frames()
	framesArray_resized, framesArray = extractKF.extractFrames(VIDEO_FILE_NAME, FRAME_PER_SECONDS, RESIZED_IMAGE_SIZE)
	clusterKF = Image_Clustering(OPTIC_MIN_SAMPLES)
	selectedImageIndex = clusterKF.clust(framesArray_resized)

	for i in selectedImageIndex:
		result = Image.fromarray(framesArray[i])
		result.save(TARGET_IMAGES_DIR + str(i).zfill(6)+".png")
	#print("--- execution time: %.2f seconds ---" % (time.time() - start_time))
	

if __name__ == '__main__':
	warnings.simplefilter(action='ignore', category=FutureWarning)
	main()