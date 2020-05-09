import time
import numpy as np
import os
import shutil
import pyclustering

from pyclustering.cluster import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy.spatial import distance
from sklearn.decomposition import PCA

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from progressbar import ProgressBar


TARGET_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/'     # The place to put the images which you want to execute clustering
CLUSTERED_IMAGES_DIR = '/home/alyssa/TCC/Estudos/frames/CentroidImage/'
IMAGE_TYPE = '.png'
PCA_COMPONENTS = 3
K = 20


class Image_Clustering:

	def main(self):
		self.label_images()

	def label_images(self):
		#print('Label images...')

		# Load a model
		model = VGG16(weights='imagenet', include_top=False)

		# Get images
		images = [f for f in os.listdir(TARGET_IMAGES_DIR) if f[-4:] in [IMAGE_TYPE]]
		assert(len(images)>0)

		X = []
		#print("Extractig features of images")
		pb = ProgressBar(len(images)).start()
		for i in range(len(images)):
			# Extract image features
			feat = self.__feature_extraction(model, TARGET_IMAGES_DIR+images[i])
			X.append(feat)
			pb.update(i)  # Update progressbar

		#print ('')
		#print("Clustering")

		# Principal Component Analysis for Dimensionality Reduction
		X = np.array(X)
		pca = PCA(n_components = PCA_COMPONENTS)
		pca.fit(X)
		X_pca= pca.transform(X)

		# Initialize Kmeans++ and execute
		initial_centers = kmeans_plusplus_initializer(X_pca, K).initialize()
		km = kmeans.kmeans(X_pca, initial_centers)
		km.process()
		clusters = km.get_clusters()
		centers = km.get_centers()

		centerIdx = list()
		for idx, p in enumerate(centers):
			#check index of the closest point from center of cluster
			centerIdx.append(np.where(X_pca == self.closest_node(p, X_pca))[0][0])

		#if PCA_COMPONENTS < 4:
			#ax = pyclustering.utils.draw_clusters(data=X_pca, clusters=clusters)

		if os.path.exists(CLUSTERED_IMAGES_DIR):
				 shutil.rmtree(CLUSTERED_IMAGES_DIR)
		if not os.path.exists(CLUSTERED_IMAGES_DIR):
			os.makedirs(CLUSTERED_IMAGES_DIR)

		label = 0

		for c in clusters:
			for item in c:
				if (item == centerIdx[label]):
					shutil.copyfile(TARGET_IMAGES_DIR+str(images[item]), CLUSTERED_IMAGES_DIR + str(label)  + IMAGE_TYPE)
					label = label+1
					#print('Save', CLUSTERED_IMAGES_DIR + str(label) + "." + IMAGE_TYPE)
					break


	def __feature_extraction(self, model, img_path):
		img = image.load_img(img_path, target_size=(224, 224))  # resize
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)  # add a dimention of samples
		x = preprocess_input(x)  # RGB 2 BGR and zero-centering by mean pixel based on the position of channels
		feat = model.predict(x)  # Get image features
		feat = feat.flatten()  # Convert 3-dimentional matrix to (1, n) array
		return feat

	def closest_node(self, node, nodes):
		closest_index = distance.cdist([node], nodes).argmin()
		return nodes[closest_index]

if __name__ == "__main__":
	start_time = time.time()
	Image_Clustering().main()
	print("--- execution time: %.2f seconds ---" % (time.time() - start_time))
