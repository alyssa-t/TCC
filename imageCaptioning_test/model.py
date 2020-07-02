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
		#hidden_with_time_axis = tf.expand_dims(hidden, 1)

		# score shape == (batch_size, 64, hidden_size)
		score = tf.nn.tanh(self.W1(features) + self.W2(hidden))

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
		self.units = 1
		# shape after fc == (batch_size, 64, embedding_dim)
		self.fc = tf.keras.layers.Dense(embedding_dim)

	def call(self, x):
		x = self.fc(x)
		x = tf.nn.relu(x)
		return x

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))

class LTSM_decoder(tf.keras.Model):
	# Since you have already extracted the features and dumped it using pickle
	# This encoder passes those features through a Fully connected layer
	def __init__(self, embedding_dim, units, vocab_size):
		super(LTSM_decoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.avgPool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")
		self.topDown_lstm = tf.keras.layers.LSTM(self.units,
									return_sequences=True,
									return_state=True,)
		self.language_lstm = tf.keras.layers.LSTM(self.units,
									return_sequences=True,
									return_state=True,)
		self.attention = BahdanauAttention(self.units)
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size, activation='softmax')

	def call(self, x, features, hidden):
		#hidden = h2_{t-1}
		#hidden : [min[imgnum,batchsize], units (512)]
		#features : [min[imgnum,batchsize], 64, embedding_dim(256)]
		# defining attention as a separate model (call parameters)
		print("HIDDEN")
		print(hidden)
		print("FEATURES")
		print(features)
		#features_with_step [min[imgnum,batchsize], 64, embedding_dim(256), 1]
		features_with_step = tf.expand_dims(features, -1)
		print(features_with_step)
		pool_feature = self.avgPool(features_with_step)
		pool_feature = tf.squeeze(pool_feature, axis=-1)
		pool_feature = tf.reshape(pool_feature, [x.shape[0],-1])
		print("POOLED FEATURE")
		print(pool_feature)
		print("X BEFORE EMBEDDING")
		print(x)
		x = self.embedding(x)
		print("X AFTER EMBEDDING BEFORE CONCAT")
		print(x)
		x = tf.concat([tf.expand_dims(hidden, 1), tf.expand_dims(pool_feature, 1), x], axis=-1)
		print("X AFTER CONCAT BEFORE LSTM")
		print(x)
		#x = tf.concat ([tf.expand_dims(context_vector, 1)])
		output1, state_h1, _ = self.topDown_lstm(x)
		print("OUTPUT1")
		print(output1)
		context_vector, attention_weights = self.attention(features, output1)
		print("CONTEXVECTOR")
		print(context_vector)
		x = tf.concat([tf.expand_dims(context_vector, 1), output1], axis=-1)
		print("X AFTER CONCAT CONTEX+OUTPUT1")
		print(x)
		output2, state_h2, _ = self.language_lstm(x)
		print("OUTPUT2")
		print(output2)
		x = self.fc1(output2)
		print("X AFTER FC1")
		print(x)
		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))
		print("X AFTER RESHAPE")
		print(x)
		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)
		print("X AFTER FC2")
		print(x)
		print("STATE2, should be equal to hidden size")
		print(state_h2)
		return x, state_h2, attention_weights
		# x shape after passing through embedding == (batch_size, 1, embedding_dim)



	def reset_state(self, batch_size):
		print("UNITS NA CLASSE")
		print(self.units)
		return tf.zeros((batch_size, self.units))



class RNN_Decoder(tf.keras.Model):
	def __init__(self, embedding_dim, units, vocab_size):
		super(RNN_Decoder, self).__init__()
		self.units = units
		#transfor each word of dictionary of size topk+1 into a fixed size vector of dimension embedding_dim
		#more info: https://qiita.com/9ryuuuuu/items/e4ee171079ffa4b87424
		#vocab_size = inputdim, embedding_dim = outputdim
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		#Gated Recurrent Unit.
		self.lstm = tf.keras.layers.GRU(self.units,
									return_sequences=True,
									return_state=True,
									recurrent_initializer='glorot_uniform')
		self.fc1 = tf.keras.layers.Dense(self.units)
		self.fc2 = tf.keras.layers.Dense(vocab_size, activation='softmax')
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
		output, state_h, state_c = self.lstm(x)

		# shape == (batch_size, max_length, hidden_size)
		x = self.fc1(output)

		# x shape == (batch_size * max_length, hidden_size)
		x = tf.reshape(x, (-1, x.shape[2]))

		# output shape == (batch_size * max_length, vocab)
		x = self.fc2(x)

		return x, state, attention_weights

	def reset_state(self, batch_size):
		return tf.zeros((batch_size, self.units))