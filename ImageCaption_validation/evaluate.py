import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def evaluate(image, max_length, attention_features_shape, decoder, encoder, tokenizer):
    hidden = decoder.reset_state(batch_size=1)
    while True:
        try:
            img_tensor_val = np.load(image)
        except:
            return []
        else:
            break
    img_tensor_val = np.load(image)
    img_tensor_val = tf.expand_dims(img_tensor_val, 0)
    features = encoder(img_tensor_val)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])
        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    print("RESULT = " + str(len_result))
    for l in range(len_result):

        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)


    plt.tight_layout()
    plt.show()