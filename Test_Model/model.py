import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		#units = dimensionality of the output space
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
		#self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		#x = self.fc(x)
		#x = tf.nn.relu(x)
		return x


class LTSM_decoder(tf.keras.Model):
	# Since you have already extracted the features and dumped it using pickle
	# This encoder passes those features through a Fully connected layer
	def __init__(self, embedding_dim, units, vocab_size):
		super(LTSM_decoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.topDown_lstm = tf.keras.layers.LSTM(self.units,
									return_sequences=True,
									return_state=True,)
		self.language_lstm = tf.keras.layers.LSTM(self.units,
									return_sequences=True,
									return_state=True,)
		self.attention = BahdanauAttention(self.units)
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, x, vi, vbar, hidden):
		#hidden = h2_{t-1}
		#hidden : [min[imgnum,batchsize], units (512)]
		#vi : [min[imgnum,batchsize], 64, embedding_dim(256)]
		# defining attention as a separate model (call parameters)
		x = self.embedding(x)

		x = tf.concat([tf.expand_dims(hidden, 1), tf.expand_dims(vbar, 1), x], axis=-1)

		output1, state_h1, _ = self.topDown_lstm(x)

		context_vector, attention_weights = self.attention(vi, output1)

		x = tf.concat([tf.expand_dims(context_vector, 1), output1], axis=-1)

		output2, state_h2, _ = self.language_lstm(x)

		x = self.fc1(output2)

		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))

		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)

		return x, state_h2, attention_weights
		# x shape after passing through embedding == (batch_size, 1, embedding_dim)



	def reset_state(self, batch_size):
		#print("UNITS NA CLASSE")
		#print(self.units)
		return tf.zeros((batch_size, self.units))

class RNN_Decoder(tf.keras.Model):
	#embedding_dim, units, vocab_size
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units

		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
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
		output, state = self.gru(x)

		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)

		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))

		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)

		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))