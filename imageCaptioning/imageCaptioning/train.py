import tensorflow as tf


def loss_function(real, pred, loss_object):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)

#@ = decorator
# more info: https://keymaso.com/programemory/python/decorator/
@tf.function
def train_step(img_tensor, target, encoder, decoder, loss_object, preprocessedTrain, optimizer):
	loss = 0

	# initializing the hidden state for each batch
	# because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])

	dec_input = tf.expand_dims([preprocessedTrain.tokenizer.word_index['<start>']] * target.shape[0], 1)

	with tf.GradientTape() as tape:
		features = encoder(img_tensor)

		for i in range(1, target.shape[1]):
		# passing the features through the decoder
			predictions, hidden, _ = decoder(dec_input, features, hidden)

			loss += loss_function(target[:, i], predictions, loss_object)

			# using teacher forcing
			dec_input = tf.expand_dims(target[:, i], 1)

	total_loss = (loss / int(target.shape[1]))

	trainable_variables = encoder.trainable_variables + decoder.trainable_variables

	gradients = tape.gradient(loss, trainable_variables)

	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss