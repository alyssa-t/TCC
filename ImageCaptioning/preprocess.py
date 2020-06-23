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

ANNOTATION_FILE = "captions_train2017_pt.txt"
ANNOTATION_FILE_PATH = "/home/alyssa/TCC/"
IMAGE_FOLDER_PATH = "/home/alyssa/TCC/train2017/"
BATCH_SIZE_INCEPTIONV3 = 16

#return a vector with captions and path+name of images
def loadAnnotation(filePath = ANNOTATION_FILE_PATH + ANNOTATION_FILE):
	with open(filePath, 'r') as f:
		annotations = json.load(f)

	# initialize vectors
	all_captions = []
	all_img_name_vector = []

	#for all annotation, associate caption with respective image path (path+imageid.jpg)
	for annot in annotations['annotations']:
		caption = '<start> ' + annot['caption'] + ' <end>'
		image_id = annot['image_id']
		full_coco_image_path = IMAGE_FOLDER_PATH + '%012d.jpg' % (image_id)

		all_img_name_vector.append(full_coco_image_path)
		all_captions.append(caption)

	# Shuffle captions and image_names with same order. Pass an int for reproducible results across multiple function calls.
	# Set a random state
	train_captions, img_name_vector = shuffle(all_captions, all_img_name_vector, random_state=1)
	return train_captions, img_name_vector

#function for limit quantity of captions to be used for training
def selectNumCaptions(train_captions, img_name_vector, numExamples = 30000):
	train_captions = train_captions[:num_examples]
	img_name_vector = img_name_vector[:num_examples]
	return train_captions, img_name_vector

# function to preprocess image for inception v3
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    #https://cloud.google.com/tpu/docs/inception-v3-advanced#preprocessing_stage
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

#return inception model withou the last fc layer
def inceptionModel ():
	#load model and Imagenet weights from keras without last fc (include_top=False)
	image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
	#define input of model
	new_input = image_model.input
	hidden_layer = image_model.layers[-1].output
	image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
	return image_features_extract_model


#Caching the features extracted from InceptionV3
#not necessary if imgXXXX.jpg.npy exists.
def catchFeatureFromInceptionV3(img_name_vector):
	# sorted = sort; set = unique element
	encode_train = sorted(set(img_name_vector))
	# Create dataset object from img_name_vector
	image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
	# transform it into a new Dataset using .map(). Dataset = (image, imagePath)
	image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE_INCEPTIONV3)

	for img, path in image_dataset:
		#for each image in image_dataset, extract features (batchsize, 8, 8,2048)
		batch_features = image_features_extract_model(img)
		#resize it to (batchsize, 64, 2048)
		batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

	for bf, p in zip(batch_features, path):
		path_of_feature = p.numpy().decode("utf-8")
		#save feature to "imgXXXXXX.jpg.npy"
		np.save(path_of_feature, bf.numpy())

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def tokenizeCaptions(train_captions):
	#k most used words
	top_k = 5000
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
	max_length = calc_max_length(train_seqs)


