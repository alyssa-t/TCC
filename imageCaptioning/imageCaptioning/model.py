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
		self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x

class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units
		#transfor each word of dictionary of size topk+1 into a fixed size vector of dimension embedding_dim
		#more info: https://qiita.com/9ryuuuuu/items/e4ee171079ffa4b87424
		#vocab_size = inputdim, embedding_dim = outputdim
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		#Gated Recurrent Unit.
		self.gru = tf.keras.layers.GRU(self.units,
									return_sequences=True,
									return_state=True,
									recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size)
		#(init parameters: object initiated, not called)
		self.attention = BahdanauAttention(self.units)

	#embedding_dim, units, vocab_size
	def call(self, x, features, hidden):
		# defining attention as a separate model (call parameters)
		context_vector, attention_weights = self.attention(features, hidden)

		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)

		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		# mmre info in: https://qiita.com/cfiken/items/04925d4da39e1a24114e
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