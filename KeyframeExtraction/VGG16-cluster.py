from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing import image

from sklearn.cluster import OPTICS, cluster_optics_dbscan
import numpy as np
import pandas as pd
import sys
import cv2
import os
import keras
from progressbar import ProgressBar
import shutil
import time


TARGET_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/'     # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/CentroidImage/'   # The place to put the images which are clustered
IMAGE_LABEL_FILE ='image_label.csv'                  # Image name and its label
INITIAL_XMEANS_CENTERS = 3

class Image_Clustering:
	def __init__(self, n_clusters=50):
		self.n_clusters = n_clusters            # The number of cluster



	def main(self):
		self.label_images()
		self.classify_images()

	def label_images(self):
		print('Label images...')

		# Load a model
		model = VGG16(weights='imagenet', include_top=False)

		# Get images
		images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in ['.png', '.jpg']]
		assert(len(images)>0)

		X = []
		pb = ProgressBar(len(images)).start()
		for i in range(len(images)):
			# Extract image features
			feat = self.__feature_extraction(model, TARGET_IMAGES_DIR+images[i])
			X.append(feat)
			pb.update(i)  # Update progressbar

		# Clutering images by OPTICS
		X = np.array(X)
		op = OPTICS(min_samples=3).fit(X)
		print('')
		print('labels:')
		print(op.labels_)
		print('')

		# Merge images and labels
		df = pd.DataFrame({'image': images, 'label': op.labels_})
		df.to_csv(IMAGE_LABEL_FILE, index=False)


	def __feature_extraction(self, model, img_path):
		img = image.load_img(img_path, target_size=(224, 224))  # resize
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)  # add a dimention of samples
		x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels

		feat = model.predict(x)  # Get image features
		feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array

		return feat


	def classify_images(self):
		print('Classify images...')

		# Get labels and images
		df = pd.read_csv(IMAGE_LABEL_FILE)
		labels = list(set(df['label'].values))

		# Delete images which were clustered before
		if os.path.exists(CLUSTERED_IMAGES_DIR):
			shutil.rmtree(CLUSTERED_IMAGES_DIR)
		if not os.path.exists(CLUSTERED_IMAGES_DIR):
			os.makedirs(CLUSTERED_IMAGES_DIR)

		for label in labels:
			if label != -1:
				print('Copy and paste label %s images.' % label)

			# Make directories named each label
			#new_dir = CLUSTERED_IMAGES_DIR + str(label) + '/'
			#if not os.path.exists(new_dir):
				#os.makedirs(new_dir)

			# Copy images to the directories
				clustered_images = df[df['label']==label]['image'].values
				for ci in clustered_images:
					src = TARGET_IMAGES_DIR + ci
					dst = CLUSTERED_IMAGES_DIR + ci
					shutil.copyfile(src, dst)
					break


if __name__ == "__main__":
	start_time = time.time()
	Image_Clustering().main()
	print("--- %s seconds ---" % (time.time() - start_time))
