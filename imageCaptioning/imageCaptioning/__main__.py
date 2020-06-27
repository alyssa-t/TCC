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
from .imageCaptioningDataset import Methods_Dataset
from .model import *
from .train import *
from .evaluate import *
from .preprocess import *

#import preprocess

ANNOTATION_FILE = "captions_train2017_pt.txt"
ANNOTATION_FILE_PATH = "/home/alyssa/TCC/"
TRAIN_IMAGE_FOLDER_PATH = "/home/alyssa/TCC/train2017/"
VALIDATION_IMAGE_FOLDER_PATH = "/home/alyssa/TCC/val2017"
BATCH_SIZE_INCEPTIONV3 = 16
NUM_CAPTIONS = 10
EXTRACT_NPY = False

BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 20
TOP_K = 5000

CHECKPOINT_PATH = "./checkpoints/train"
TEST_IMAGE_PATH = "/home/alyssa/Pictures/1.png"

# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
#embeddingdim = vector dimension for word mapping
embedding_dim = 256
#units = dimensionality of the output space of models
units = 512
# words vector size
vocab_size = TOP_K + 1
#num_setps = int division NUM=CAPTIONS/BATCH_SIZE. ex: 4//1.5 = 2
num_steps = NUM_CAPTIONS // BATCH_SIZE




def main():

    #Code for setup GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    #Load images and captions
    trainCaptions, img_name_vector = loadAnnotation(ANNOTATION_FILE_PATH+ANNOTATION_FILE,
                                                            TRAIN_IMAGE_FOLDER_PATH,
                                                            NUM_CAPTIONS)
    #Create model for extract features using InceptionV3
    image_features_extract_model = createInceptionModel()
    if (EXTRACT_NPY):
        print("extracting feature of training set")
        createNpyFile (img_name_vector,
                        image_features_extract_model,
                        BATCH_SIZE_INCEPTIONV3)
    #Tokenize captions
    cap_vector, maxLength, tokenizer = tokenizeCaptions(trainCaptions, TOP_K)

    #Create Dataset
    methodDataset = Methods_Dataset(img_name_vector,
                                    cap_vector,
                                    BATCH_SIZE,
                                    BUFFER_SIZE)

    dataset = methodDataset.createDataset()

    #Create encoder decoder model
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    #Define Optimizer and loss_object
    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    #Define Checkpoints
    ckpt = tf.train.Checkpoint(encoder=encoder,
                                decoder=decoder,
                                optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    #Load Checkpoint if exists
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in CHECKPOINT_PATH
        ckpt.restore(ckpt_manager.latest_checkpoint)

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []
    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor,
                                                target,
                                                encoder,
                                                decoder,
                                                loss_object,
                                                tokenizer,
                                                optimizer)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                    epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                                total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
    
    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()
 
    result, attention_plot = evaluate(TEST_IMAGE_PATH,
                                        maxLength,
                                        attention_features_shape,
                                        decoder,
                                        encoder,
                                        image_features_extract_model,
                                        tokenizer)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(TEST_IMAGE_PATH, result, attention_plot)


if __name__ == '__main__':
    main()
