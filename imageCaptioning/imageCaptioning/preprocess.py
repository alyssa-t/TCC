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
from glob import glob
from PIL import Image
from tqdm import tqdm

class Methods_preprocess:
	#return a vector of captions and path+name of images
	def loadAnnotation(self, annotationFilePath, imageFilePath):
		with open(annotationFilePath, 'r') as f:
			annotations = json.load(f)

		# initialize vectors
		all_captions = []
		all_img_name_vector = []

		#for all annotation, associate caption with respective image path (path+imageid.jpg)
		for annot in annotations['annotations']:
			caption = '<start> ' + annot['caption'] + ' <end>'
			image_id = annot['image_id']
			full_coco_image_path = imageFilePath + '%012d.jpg' % (image_id)

			all_img_name_vector.append(full_coco_image_path)
			all_captions.append(caption)

		# Shuffle captions and image_names with same order. Pass an int for reproducible results across multiple function calls.
		# Set a random state
		train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)
		return train_captions, img_name_vector

	#function for limit quantity of captions to be used for training
	#return a vector of captions and path+name of images
	def selectNumCaptions(self, train_captions, img_name_vector, numExamples = 30000):
		train_captions = train_captions[:numExamples]
		img_name_vector = img_name_vector[:numExamples]
		return train_captions, img_name_vector

	# function to preprocess image for inception v3
	#return a vector of 299x299 images and file path
	def load_image(self, image_path):
		img = tf.io.read_file(image_path)
		img = tf.image.decode_jpeg(img, channels=3)
		img = tf.image.resize(img, (299, 299))
		#https://cloud.google.com/tpu/docs/inception-v3-advanced#preprocessing_stage
		img = tf.keras.applications.inception_v3.preprocess_input(img)
		return img, image_path

	#return inception model withou the last fc layer
	def createInceptionModel (self):
		#load model and Imagenet weights from keras without last fc (include_top=False)
		image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
		#define input of model
		new_input = image_model.input
		hidden_layer = image_model.layers[-1].output
		image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
		return image_features_extract_model

	#Caching the features extracted from InceptionV3
	#not necessary if imgXXXX.jpg.npy already exists.
	def catchFeatureFromInceptionV3(self,img_name_vector, image_features_extract_model, batchSize, extract_npy):
		# sorted = sort; set = unique element
		encode_train = sorted(set(img_name_vector))
		# Create dataset object from img_name_vector
		image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
		# transform it into a new Dataset using .map(). Dataset = (image, imagePath)
		image_dataset = image_dataset.map(self.load_image, 
											num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batchSize)

		if (extract_npy):
			for img, path in tqdm(image_dataset):
				#for each image in image_dataset, extract features (batchsize, 8, 8,2048)
				batch_features = image_features_extract_model(img)
				#resize it to (batchsize, 64, 2048)
				batch_features = tf.reshape(batch_features,
									(batch_features.shape[0], -1, batch_features.shape[3]))

			for bf, p in zip(batch_features, path):
				path_of_feature = p.numpy().decode("utf-8")
				#save feature to "imgXXXXXX.jpg.npy"
				np.save(path_of_feature, bf.numpy())

	def calc_max_length(self,tensor):
		return max(len(t) for t in tensor)

	def tokenizeCaptions(self,train_captions, top_k):
		#k most used words
		#create a tokenizer. oov=out-of-vocabulary are cosidered "unknown" and special characters are ignored 
		tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
													oov_token="<unk>",
													filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
		#Updates internal vocabulary based on a list of texts.
		tokenizer.fit_on_texts(train_captions)
		#Transforms each text in texts in a sequence of integers.
		train_seqs = tokenizer.texts_to_sequences(train_captions)

		#consider 0 as padding
		tokenizer.word_index['<pad>'] = 0
		tokenizer.index_word[0] = '<pad>'

		# Create the tokenized vectors
		train_seqs = tokenizer.texts_to_sequences(train_captions)

		# Pad each vector to the max_length of the captions
		# If you do not provide a max_length value, pad_sequences calculates it automatically
		#if argument value is not provided, is padded with 0 = '<pad>'
		cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

		# Calculates the max_length, which is used to store the attention weights
		max_length = self.calc_max_length(train_seqs)

		return cap_vector, max_length, tokenizer

class PreprocessImageAndCaptions(Methods_preprocess):
	def __init__(self, annotationFilePath, imageFilePath, batchSizeInceptionV3):
		super(PreprocessImageAndCaptions, self).__init__()
		self.annotationFilePath = annotationFilePath
		self.imageFilePath = imageFilePath
		self.batchSizeInceptionV3 = batchSizeInceptionV3
		self.trainCaptions = None
		self.img_name_vector = None
		self.cap_vector = None
		self.maxLength = None
		self.tokenizer = None

	def createNpyFile(self, numExamples = None, extract_npy = False):
		self.trainCaptions, self.img_name_vector = self.loadAnnotation(self.annotationFilePath, 
                                                            self.imageFilePath)
		if (numExamples is not None):
			self.trainCaptions, self.img_name_vector = self.selectNumCaptions (self.trainCaptions, self.img_name_vector, numExamples)
		image_features_extract_model = self.createInceptionModel()
		self.catchFeatureFromInceptionV3(self.img_name_vector, 
											image_features_extract_model,
											self.batchSizeInceptionV3,
											extract_npy)

	def createCaptionFile(self, top_k=5000):
		self.cap_vector, self.maxLength, self.tokenizer = self.tokenizeCaptions(self.trainCaptions, top_k)