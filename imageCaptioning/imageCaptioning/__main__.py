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
from .preprocess import PreprocessImageAndCaptions
from .imageCaptioningDataset import Methods_Dataset
from .model import *
from .train import *

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
TOP_K = 5000

CHECKPOINT_PATH = "/home/alyssa/TCC/TCC/imageCaptioning/imageCaptioning/checkpoints/train"

embedding_dim = 256
units = 512
vocab_size = TOP_K + 1
num_steps = NUM_CAPTIONS // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

def main():

    #code for setup GPU
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    ####################

    print("extracting feature of training set")
    preprocessedTrain = PreprocessImageAndCaptions(ANNOTATION_FILE_PATH+ANNOTATION_FILE,
                                                    TRAIN_IMAGE_FOLDER_PATH,
                                                    BATCH_SIZE_INCEPTIONV3)
    preprocessedTrain.createNpyFile(NUM_CAPTIONS, EXTRACT_NPY)
    preprocessedTrain.createCaptionFile(TOP_K)

    methodDataset = Methods_Dataset(preprocessedTrain.img_name_vector, 
                    preprocessedTrain.cap_vector,
                    BATCH_SIZE,
                    BUFFER_SIZE)

    dataset = methodDataset.createDataset()

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    ckpt = tf.train.Checkpoint(encoder=encoder,
                                decoder=decoder,
                                optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)

    start_epoch = 0

    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in CHECKPOINT_PATH
        ckpt.restore(ckpt_manager.latest_checkpoint)

    loss_plot = []

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset

    EPOCHS = 20

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, loss_object, preprocessedTrain, optimizer)
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


if __name__ == '__main__':
    main()
