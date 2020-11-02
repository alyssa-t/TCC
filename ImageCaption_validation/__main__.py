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
from .model import *
from .evaluate import *
from .preprocess import *
from .telegram_logger import telegramSendMessage

ANNOTATION_FILE = "captions_train2017_pt.txt"
VALIDATION_IMAGE_FOLDER_PATH = "/home/alyssa/Desktop/projeto_graduacao/Test/testnpy/"
CHECKPOINT_PATH = "/home/alyssa/Desktop/projeto_graduacao/checkpoints/train/2-8-yolo-v.4"
TOKENIZER_PATH = "/home/alyssa/Desktop/projeto_graduacao/"
TEST_IMAGE_PATH = "*.npy"
TOP_K = 10000
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64
#embeddingdim = vector dimension for word mapping
embedding_dim = 2048
#units = dimensionality of the output space of models
units = 2048
# words vector size
vocab_size = TOP_K + 1
maxLength = 15


def main():

    #Code for setup GPU
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    tf.compat.v1.enable_eager_execution()
    

    #Create encoder decoder model
    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    #load tokenizer
    with open(TOKENIZER_PATH+'tokenizer.pickle', 'rb') as handle:
    	tokenizer = pickle.load(handle)

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
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])*5
        # restoring the latest checkpoint in CHECKPOINT_PATH
        status = ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print("Restored CheckPoint from" + ckpt_manager.directory)
        status.assert_existing_objects_matched()
    else:
        raise ValueError('No checkpoint detected.') 

    print("======= VAL IMAGES =======")
    for npyfile in glob.glob(VALIDATION_IMAGE_FOLDER_PATH + "*.npy"):
        print()
        print("Captioning image :")
        print(npyfile)
        print ("File:" + str(npyfile))
        for i in range(10):
            result = evaluate(npyfile,
                                                maxLength,
                                                attention_features_shape,
                                                decoder,
                                                encoder,
                                                tokenizer)
            print ('Prediction Caption:', ' '.join(result))  

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    telegramSendMessage("loading captions")
    main()
    """
    try:
        main()

    except Exception as e:
        print(e)
        telegramSendMessage('Houve um erro de execução')
        telegramSendMessage(str(e))
    """
    
