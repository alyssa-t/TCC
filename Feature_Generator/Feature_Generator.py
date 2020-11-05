import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import skimage.measure
import cv2
import numpy as np
import os
import shutil

from glob import glob
from PIL import Image
from tqdm import tqdm


class Feature_Generator:
	def __init__(self, YOLO_RESULT_FILE, PATH_TMP_FOLDER, PATH_NPY_FILES, BATCH_SIZE_INCEPTIONV3, NTH_SLASH):
		self.yoloResultFile = YOLO_RESULT_FILE
		self.pathTmpFolder = PATH_TMP_FOLDER
		self.pathNpyFiles = PATH_NPY_FILES
		self.batchSize = BATCH_SIZE_INCEPTIONV3
		self.nthSlash = NTH_SLASH


	def sortKey(self, elem):
		return elem[2]

	def load_image(self, image_path):
		img = tf.io.read_file(image_path)
		img = tf.image.decode_jpeg(img, channels=3)
		img = tf.image.resize(img, (299, 299))
		#https://cloud.google.com/tpu/docs/inception-v3-advanced#preprocessing_stage
		img = tf.keras.applications.inception_v3.preprocess_input(img)
		return img, image_path
	
	def cropImage(self):
		list_prob = []
		threshold = 70
		conseq = False

		#
		count = 0
		#
		# take second element for sort
		#create folders to store data
		with open(self.yoloResultFile, "r") as result_file:
			for line in result_file:
				if (len(line)<2) or ("seen 64" in line):
					continue

				words = line.split()
				#save image path
				if (words[0][0] == "/"):
					#fim da lista
					if (conseq == True):
						original_image.save(self.pathTmpFolder + "_" + path.split("/")[self.nthSlash].split(".")[0]+"_part" + str(0)+".jpg")
						conseq = False

					elif (len(list_prob)!=0):
						list_prob.sort(reverse=True)
						if (len(list_prob) >= 3):
							list_prob = list_prob[:3]
							if (list_prob[2][0] < threshold):
								list_prob = list_prob[:2]
						list_prob.sort(key=self.sortKey, reverse=True)

						for idx in range (len(list_prob)):
							crop_img = original_image.crop(list_prob[idx][1])
							crop_img.save(self.pathTmpFolder + "_" + path.split("/")[self.nthSlash].split(".")[0]+"_part" + str(idx)+".jpg")

						list_prob = []
						conseq = True
					path = words[0][0:-1]
					original_image = cv2.imread(path)
					original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
					original_image = Image.fromarray(original_image.astype((np.uint8)))
					


				#crop image
				if (words[-1][-1] == ")" ):
					#append list of prob with respective area 
					area = ( int(words[-7]), int(words[-5]), int(words[-3])+int(words[-7]), int(words[-1][:-1])+int(words[-5]))
					size = int(words[-1][:-1])*int(words[-3])
					list_prob.append([int(words[-9][:-1]), area, size])
					conseq = False
					#prepare image for VGG
					#crop_img.show()
					
					continue
			original_image.save(self.pathTmpFolder + "_" + path.split("/")[self.nthSlash].split(".")[0]+"_part" + str(0)+".jpg")
			conseq = False

	def extractFeature(self):
		# Load all jpg file
		original_names = []
		for name in glob(self.pathTmpFolder + "*.jpg"):
			original_names.append(name)

		image_model = tf.keras.applications.InceptionV3(include_top=False,
														weights='imagenet')
		new_input = image_model.input
		hidden_layer = image_model.layers[-1].output

		image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

		# Get unique images
		encode_train = sorted(set(original_names))

		# Feel free to change batch_size according to your system configuration
		image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
		image_dataset = image_dataset.map(
			self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batchSize)

		for img, path in tqdm(image_dataset):
			batch_features = image_features_extract_model(img)
			batch_features = tf.reshape(batch_features,
										(batch_features.shape[0], -1, batch_features.shape[3]))

			for bf, p in zip(batch_features, path):
				path_of_feature = p.numpy().decode("utf-8")
				np.save(path_of_feature, bf.numpy())

		list_ids = []
		original_imageName = glob(self.pathTmpFolder + "*.npy")

		for img_name in original_imageName:
			list_ids.append(img_name.split("_")[3])

		#list of all list_ids in a file
		list_ids = list(set(list_ids))
		#print(list_ids)
		zeros = np.zeros((1, 2048))
		#for each id
		for image_id in list_ids:

			parts = glob(self.pathTmpFolder + "_"+ image_id + "*.npy")
			for idx, name in enumerate(parts):
				if idx == 0:
					img_tensor = np.load(name)
					img_tensor = skimage.measure.block_reduce(img_tensor, (2048,1 ), np.mean)
					img_tensor = np.squeeze(img_tensor, axis=0)
					img_tensor = tf.expand_dims(img_tensor, 0)
				else:
					aux = np.load(name)
					aux = skimage.measure.block_reduce(aux, (2048,1 ), np.mean)
					aux = np.squeeze(aux, axis=0)
					aux = tf.expand_dims(aux, 0)
					img_tensor = np.concatenate((img_tensor, aux),axis=0)
				os.remove(name)

			while (img_tensor.shape != (4,2048)):
				img_tensor = np.concatenate((img_tensor, zeros),axis=0)

			npyName = image_id + ".jpg"
			np.save(self.pathNpyFiles + npyName, img_tensor)