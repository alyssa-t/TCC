import tensorflow as tf

def loss_function(real, pred, loss_object):
	mask = tf.math.logical_not(tf.math.equal(real, 0))
	loss_ = loss_object(real, pred)

	mask = tf.cast(mask, dtype=loss_.dtype)
	loss_ *= mask

	return tf.reduce_mean(loss_)

#@ = decorator
# more info: https://keymaso.com/programemory/python/decorator/
#@tf.functionを付けると、その関数はグラフにコンパイルされ速く動作するそうです
@tf.function
def train_step(img_tensor, target, encoder, decoder, loss_object, tokenizer, optimizer):
	loss = 0
	# initializing the hidden state for each batch
	# because the captions are not related from image to image
	hidden = decoder.reset_state(batch_size=target.shape[0])

	dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

	#save operations to calculloss_functionate gradient later (tape)
	#more info: https://blog.shikoan.com/gradienttape-multi/
	with tf.GradientTape() as tape:
		#vi : [min[imgnum,batchsize], batchsize, embedding_dim(256)]
		vi = encoder (img_tensor)
		#vi = img_tensor
		#print("vi shape")
		#print(vi.shape)
		#input("enter.")
		for i in range(1, target.shape[1]):
			# passing the vi through the decoder
			# decoder(x, vi, hidden)
			#ignore last return value = attention_weights
			predictions, hidden, _ = decoder(dec_input, vi, hidden)
			#predictions, hidden, _ = decoder(dec_input, vi, vbar, hidden)

			#predicted_id = tf.random.categorical(predictions, 1)[0][0]

			loss += loss_function(target[:, i], predictions, loss_object)

			# using teacher forcing
			dec_input = tf.expand_dims(target[:, i], 1)


	total_loss = (loss / int(target.shape[1]))

	trainable_variables = encoder.trainable_variables + decoder.trainable_variables

	#tape.gradient(dy,dx)
	gradients = tape.gradient(loss, trainable_variables)

	#apply gradient qith optimizer (ex.adam)
	optimizer.apply_gradients(zip(gradients, trainable_variables))

	return loss, total_loss