# -*- coding: utf-8 -*-
#from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
import warnings
from glob import glob
from PIL import Image
import pickle
from tqdm import tqdm
from telegram_logger import telegramSendMessage

warnings.simplefilter(action='ignore', category=FutureWarning)
try:

  #gpu_devices = tf.config.experimental.list_physical_devices('GPU')
  os.environ["CUDA_VISIBLE_DEVICES"]="0"

  #Absolute path
  #Absolute path
  ANNOTATION_FILE = "captions_train2017.json"
  ANNOTATION_FILE_PATH = "*/"
  IMAGE_FOLDER_PATH = "*/"

  BATCH_SIZE_INCEPTIONV3 = 16

  # Download caption annotation files
  annotation_file = ANNOTATION_FILE
  

  # Read the json file
  with open(annotation_file, 'r') as f:
      annotations = json.load(f)

  # Store captions and image names in vectors
  all_captions = []
  all_img_name_vector = []

  for annot in annotations['annotations']:
      caption = '<start> ' + annot['caption'] + ' <end>'
      image_id = annot['image_id']
      full_coco_image_path = IMAGE_FOLDER_PATH + '%012d.jpg' % (image_id)

      all_img_name_vector.append(full_coco_image_path)
      all_captions.append(caption)

  # Shuffle captions and image_names together
  # Set a random state
  train_captions, img_name_vector = shuffle(all_captions,
                                            all_img_name_vector,
                                            random_state=1)

  # Select the first 30000 captions from the shuffled set
  # Pode aumentar as imagens para melhor treinamento
  num_examples = 30000
  train_captions = train_captions[:num_examples]
  img_name_vector = img_name_vector[:num_examples]

  print(len(train_captions))
  print(len(all_captions))

  # function for preprocess image for inception v3
  def load_image(image_path):
      img = tf.io.read_file(image_path)
      img = tf.image.decode_jpeg(img, channels=3)
      img = tf.image.resize(img, (299, 299))
      #https://cloud.google.com/tpu/docs/inception-v3-advanced#preprocessing_stage
      img = tf.keras.applications.inception_v3.preprocess_input(img)
      return img, image_path

  """## Initialize InceptionV3 and load the pretrained Imagenet weights

  Now you'll create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture. 
  The shape of the output of this layer is ```8x8x2048```. 
  You use the last convolutional layer because you are using attention in this example. 
  You don't perform this initialization during training because it could become a bottleneck.

  * You forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
  * After all the images are passed through the network, you pickle the dictionary and save it to disk.
  """

  image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                  weights='imagenet')
  new_input = image_model.input
  hidden_layer = image_model.layers[-1].output

  image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
  """
  # Get unique images
  encode_train = sorted(set(img_name_vector))

  # Feel free to change batch_size according to your system configuration
  image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
  image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE_INCEPTIONV3)

  for img, path in tqdm(image_dataset):
    batch_features = image_features_extract_model(img)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

    for bf, p in zip(batch_features, path):
      path_of_feature = p.numpy().decode("utf-8")
      np.save(path_of_feature, bf.numpy())

  """
  """## Preprocess and tokenize the captions

  * First, you'll tokenize the captions (for example, by splitting on spaces). 
      This gives us a  vocabulary of all of the unique words in the data (for example, "surfing", "football", and so on).
  * Next, you'll limit the vocabulary size to the top 5,000 words (to save memory). 
      You'll replace all other words with the token "UNK" (unknown).
  * You then create word-to-index and index-to-word mappings.
  * Finally, you pad all sequences to be the same length as the longest one.
  """

  # Find the maximum length of any caption in our dataset
  def calc_max_length(tensor):
      return max(len(t) for t in tensor)

  # Choose the top 5000 words from the vocabulary
  top_k = 10000
  tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
  tokenizer.fit_on_texts(train_captions)
  train_seqs = tokenizer.texts_to_sequences(train_captions)

  tokenizer.word_index['<pad>'] = 0
  tokenizer.index_word[0] = '<pad>'

  # Create the tokenized vectors
  train_seqs = tokenizer.texts_to_sequences(train_captions)

  # Pad each vector to the max_length of the captions
  # If you do not provide a max_length value, pad_sequences calculates it automatically
  cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

  # Calculates the max_length, which is used to store the attention weights
  max_length = calc_max_length(train_seqs)

  img_name_train = img_name_vector
  cap_train = cap_vector
  print(len(img_name_train))
  print(len(cap_train))
  """## Split the data into training and testing"""
  """
  # Create training and validation sets using an 80-20 split
  img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                      cap_vector,
                                                                      test_size=0.2,
                                                                      random_state=0)

  len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)
  """
  """## Create a tf.data dataset for training
  Our images and captions are ready! Next, let's create a tf.data dataset to use for training our model.
  """

  # Feel free to change these parameters according to your system's configuration

  BATCH_SIZE = 64
  BUFFER_SIZE = 1000
  embedding_dim = 256
  units = 512
  vocab_size = top_k + 1
  num_steps = len(img_name_train) // BATCH_SIZE
  # Shape of the vector extracted from InceptionV3 is (64, 2048)
  # These two variables represent that vector shape
  features_shape = 2048
  attention_features_shape = 64

  # Load the numpy files
  def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

  dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

  # Use map to load the numpy files in parallel
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

  # Shuffle and batch
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

  """## Model

  Fun fact: the decoder below is identical to the one in the example for [Neural Machine Translation with Attention](../sequences/nmt_with_attention.ipynb).

  The model architecture is inspired by the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) paper.

  * In this example, you extract the features from the lower convolutional layer of InceptionV3 giving us a vector of shape (8, 8, 2048).
  * You squash that to a shape of (64, 2048).
  * This vector is then passed through the CNN Encoder (which consists of a single Fully connected layer).
  * The RNN (here GRU) attends over the image to predict the next word.
  """

  class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
      super(BahdanauAttention, self).__init__()
      self.W1 = tf.keras.layers.Dense(units)
      self.W2 = tf.keras.layers.Dense(units)
      self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
      # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

      # hidden shape == (batch_size, hidden_size)
      # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
      hidden_with_time_axis = tf.expand_dims(hidden, 1)

      # score shape == (batch_size, 64, hidden_size)
      score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

      # attention_weights shape == (batch_size, 64, 1)
      # you get 1 at the last axis because you are applying score to self.V
      attention_weights = tf.nn.softmax(self.V(score), axis=1)

      # context_vector shape after sum == (batch_size, hidden_size)
      context_vector = attention_weights * features
      context_vector = tf.reduce_sum(context_vector, axis=1)

      return context_vector, attention_weights

  class CNN_Encoder(tf.keras.Model):
      # Since you have already extracted the features and dumped it using pickle
      # This encoder passes those features through a Fully connected layer
      def __init__(self, embedding_dim):
          super(CNN_Encoder, self).__init__()
          # shape after fc == (batch_size, 64, embedding_dim)
          self.fc = tf.keras.layers.Dense(embedding_dim)

      def call(self, x):
          x = self.fc(x)
          x = tf.nn.relu(x)
          return x

  class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
      super(RNN_Decoder, self).__init__()
      self.units = units

      self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      self.gru = tf.keras.layers.LSTM(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
      self.fc1 = tf.keras.layers.Dense(self.units)
      self.fc2 = tf.keras.layers.Dense(vocab_size)

      self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
      # defining attention as a separate model
      context_vector, attention_weights = self.attention(features, hidden)

      # x shape after passing through embedding == (batch_size, 1, embedding_dim)
      x = self.embedding(x)

      # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
      x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

      # passing the concatenated vector to the GRU
      output, state,_ = self.gru(x)

      # shape == (batch_size, max_length, hidden_size)
      x = self.fc1(output)

      # x shape == (batch_size * max_length, hidden_size)
      x = tf.reshape(x, (-1, x.shape[2]))

      # output shape == (batch_size * max_length, vocab)
      x = self.fc2(x)

      return x, state, attention_weights

    def reset_state(self, batch_size):
      return tf.zeros((batch_size, self.units))

  encoder = CNN_Encoder(embedding_dim)
  decoder = RNN_Decoder(embedding_dim, units, vocab_size)

  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')

  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

  """## Checkpoint"""

  checkpoint_path = "./checkpoints/train/ver1"
  ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  start_epoch = 0
  """
  if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint) 
  """
  """## Training

  * You extract the features stored in the respective `.npy` files and then pass those features through the encoder.
  * The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
  * The decoder returns the predictions and the decoder hidden state.
  * The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
  * Use teacher forcing to decide the next input to the decoder.
  * Teacher forcing is the technique where the target word is passed as the next input to the decoder.
  * The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
  """

  # adding this in a separate cell because if you run the training cell
  # many times, the loss_plot array will be reset
  loss_plot = []

  @tf.function
  def train_step(img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

  EPOCHS = 20

  for epoch in range(start_epoch, EPOCHS):
      start = time.time()
      total_loss = 0
      telegramSendMessage("ver 1 no comeco do epoch "+str(epoch))
      for (batch, (img_tensor, target)) in enumerate(dataset):
          batch_loss, t_loss = train_step(img_tensor, target)
          total_loss += t_loss

          if batch % 100 == 0:
              print ('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
      # storing the epoch end loss value to plot later
      loss_plot.append(total_loss / num_steps)

      #if epoch % 5 == 0:
      ckpt_manager.save()
      telegramSendMessage("ver 1 no fim do epoch "+str(epoch))
      print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                          total_loss/num_steps))
      print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

  plt.plot(loss_plot)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Loss Plot')
  plt.show()

  """## Caption!

  * The evaluate function is similar to the training loop, except you don't use teacher forcing here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
  * Stop predicting when the model predicts the end token.
  * And store the attention weights for every time step.
  """

  def evaluate(image):
      attention_plot = np.zeros((max_length, attention_features_shape))

      hidden = decoder.reset_state(batch_size=1)

      temp_input = tf.expand_dims(load_image(image)[0], 0)
      img_tensor_val = image_features_extract_model(temp_input)
      img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

      features = encoder(img_tensor_val)

      dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
      result = []

      for i in range(max_length):
          predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

          attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

          predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
          result.append(tokenizer.index_word[predicted_id])

          if tokenizer.index_word[predicted_id] == '<end>':
              return result, attention_plot

          dec_input = tf.expand_dims([predicted_id], 0)

      attention_plot = attention_plot[:len(result), :]
      return result, attention_plot

  def plot_attention(image, result, attention_plot):
      temp_image = np.array(Image.open(image))

      fig = plt.figure(figsize=(10, 10))

      len_result = len(result)
      for l in range(len_result):
          temp_att = np.resize(attention_plot[l], (8, 8))
          ax = fig.add_subplot(len_result//2, len_result//2, l+1)
          ax.set_title(result[l])
          img = ax.imshow(temp_image)
          ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

      plt.tight_layout()
      plt.show()
  """
  # captions on the validation set
  rid = np.random.randint(0, len(img_name_val))
  image = img_name_val[rid]
  real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
  result, attention_plot = evaluate(image)

  print ('Real Caption:', real_caption)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image, result, attention_plot)
  """
  """## Try it on your own images
  For fun, below we've provided a method you can use to caption your own images with the model we've just trained. Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the training data (so be prepared for weird results!)
  """

  image_url = 'https://tensorflow.org/images/surf.jpg'
  image_extension = image_url[-4:]
  image_path = tf.keras.utils.get_file('image'+image_extension,
                                      origin=image_url)

  result, attention_plot = evaluate(image_path)
  print ('Prediction Caption:', ' '.join(result))
  plot_attention(image_path, result, attention_plot)
  # opening the image
  Image.open(image_path)
  telegramSendMessage("FIM DO VER1")
except Exception as e:
  telegramSendMessage('Houve um erro de execução no ver1')
  telegramSendMessage(str(e))


"""# Next steps

Congrats! You've just trained an image captioning model with attention. Next, take a look at this example [Neural Machine Translation with Attention](../sequences/nmt_with_attention.ipynb). It uses a similar architecture to translate between Spanish and English sentences. You can also experiment with training the code in this notebook on a different dataset.
"""