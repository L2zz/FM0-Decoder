import tensorflow as tf



class Encoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim):
    super(Encoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
    self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)

  def call(self, input_features):
    activation = self.hidden_layer(input_features)
    return self.output_layer(activation)



class Decoder(tf.keras.layers.Layer):
  def __init__(self, intermediate_dim, original_dim):
    super(Decoder, self).__init__()
    self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
    self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu)

  def call(self, code):
    activation = self.hidden_layer(code)
    return self.output_layer(activation)



class Autoencoder(tf.keras.Model):
  def __init__(self, intermediate_dim, original_dim):
    super(Autoencoder, self).__init__()
    self.input_layer = tf.keras.layers.Input(shape=(6400,))
    self.encoder = Encoder(intermediate_dim=intermediate_dim)
    self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

  def call(self, input_features):
    code = self.encoder(input_features)
    reconstructed = self.decoder(code)
    return reconstructed
