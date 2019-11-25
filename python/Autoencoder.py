import tensorflow as tf
import datetime

import global_vars
tail = global_vars.tail
model_path = global_vars.model_path
model_full_path = global_vars.model_full_path
model_tail = global_vars.model_tail
log_full_path = global_vars.log_full_path



learning_rate = 0.001
learning_epoch = 50

class Autoencoder(tf.keras.Model):
  def __init__(self):
    try:
      super(Autoencoder, self).__init__()

      optimizer = tf.keras.optimizers.Adam(learning_rate)
      self.model = eval("self.build_model" + tail + model_tail)()
      self.model.compile(loss="mse", optimizer=optimizer)
      self.model.summary()

    except Exception as ex:
      print("[Autoencoder.__init__]", end=" ")
      print(ex)



  def build_model_enc256(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 256

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")
      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden1")
      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, activation=tf.nn.relu, name="hidden2")
      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.hidden_layer1(self.input_layer)
      layer = self.hidden_layer2(layer)
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256]", end=" ")
      print(ex)



  def build_model_enc256_reg(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 3100
      size_output_layer = 256
      reg = 0.000001

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")
      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(reg), name="hidden1")
      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(reg), name="hidden2")
      self.hidden_layer3 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(reg), name="hidden3")
      self.hidden_layer4 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(reg), name="hidden4")
      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.hidden_layer1(self.input_layer)
      layer = self.hidden_layer2(layer)
      layer = self.hidden_layer3(layer)
      layer = self.hidden_layer4(layer)
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_reg]", end=" ")
      print(ex)



  def build_model_enc256_batch(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 3200
      size_hidden_layer2 = 256
      size_hidden_layer3 = 3200
      size_output_layer = 256

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, name="hidden1")
      self.batch_layer1 = tf.keras.layers.BatchNormalization(name="batch1")
      self.activation_layer1 = tf.keras.layers.Activation(tf.nn.relu, name="activation1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, name="hidden2")
      self.batch_layer2 = tf.keras.layers.BatchNormalization(name="batch2")
      self.activation_layer2 = tf.keras.layers.Activation(tf.nn.relu, name="activation2")

      self.hidden_layer3 = tf.keras.layers.Dense(units=size_hidden_layer2, name="hidden3")
      self.batch_layer3 = tf.keras.layers.BatchNormalization(name="batch3")
      self.activation_layer3 = tf.keras.layers.Activation(tf.nn.relu, name="activation3")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.activation_layer1(self.batch_layer1(self.hidden_layer1(self.input_layer)))
      layer = self.activation_layer2(self.batch_layer2(self.hidden_layer2(layer)))
      layer = self.activation_layer3(self.batch_layer3(self.hidden_layer3(layer)))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_batch]", end=" ")
      print(ex)



  def build_model_enc256_dropout(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 256
      dropout_rate = 0.1

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden1")
      self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, activation=tf.nn.relu, name="hidden2")
      self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.dropout_layer1(self.hidden_layer1(self.input_layer))
      layer = self.dropout_layer2(self.hidden_layer2(layer))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_dropout]", end=" ")
      print(ex)



  def build_model_enc256_dropout_batch(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 256
      dropout_rate = 0.1

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, name="hidden1")
      self.batch_layer1 = tf.keras.layers.BatchNormalization(name="batch1")
      self.activation_layer1 = tf.keras.layers.Activation(tf.nn.relu, name="activation1")
      self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, name="hidden2")
      self.batch_layer2 = tf.keras.layers.BatchNormalization(name="batch2")
      self.activation_layer2 = tf.keras.layers.Activation(tf.nn.relu, name="activation2")
      self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.dropout_layer1(self.activation_layer1(self.batch_layer1(self.hidden_layer1(self.input_layer))))
      layer = self.dropout_layer2(self.activation_layer2(self.batch_layer2(self.hidden_layer2(layer))))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_dropout_batch]", end=" ")
      print(ex)



  def build_model_enc256_5(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 1280

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")
      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden1")
      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, activation=tf.nn.relu, name="hidden2")
      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.hidden_layer1(self.input_layer)
      layer = self.hidden_layer2(layer)
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_5]", end=" ")
      print(ex)



  def build_model_enc256_5_batch(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 1280

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, name="hidden1")
      self.batch_layer1 = tf.keras.layers.BatchNormalization(name="batch1")
      self.activation_layer1 = tf.keras.layers.Activation(tf.nn.relu, name="activation1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, name="hidden2")
      self.batch_layer2 = tf.keras.layers.BatchNormalization(name="batch2")
      self.activation_layer2 = tf.keras.layers.Activation(tf.nn.relu, name="activation2")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.activation_layer1(self.batch_layer1(self.hidden_layer1(self.input_layer)))
      layer = self.activation_layer2(self.batch_layer2(self.hidden_layer2(layer)))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_5_batch]", end=" ")
      print(ex)



  def build_model_enc256_5_dropout(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 1280
      dropout_rate = 0.1

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden1")
      self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, activation=tf.nn.relu, name="hidden2")
      self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.dropout_layer1(self.hidden_layer1(self.input_layer))
      layer = self.dropout_layer2(self.hidden_layer2(layer))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_5_dropout]", end=" ")
      print(ex)



  def build_model_enc256_5_dropout_batch(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 256
      size_hidden_layer2 = 3200
      size_output_layer = 1280
      dropout_rate = 0.1

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, name="hidden1")
      self.batch_layer1 = tf.keras.layers.BatchNormalization(name="batch1")
      self.activation_layer1 = tf.keras.layers.Activation(tf.nn.relu, name="activation1")
      self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, name="hidden2")
      self.batch_layer2 = tf.keras.layers.BatchNormalization(name="batch2")
      self.activation_layer2 = tf.keras.layers.Activation(tf.nn.relu, name="activation2")
      self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.dropout_layer1(self.activation_layer1(self.batch_layer1(self.hidden_layer1(self.input_layer))))
      layer = self.dropout_layer2(self.activation_layer2(self.batch_layer2(self.hidden_layer2(layer))))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256_5_dropout_batch]", end=" ")
      print(ex)



  def build_model_enc128(self):
    try:
      size_input_layer = 6400
      size_hidden_layer = 3100
      size_output_layer = 128

      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")
      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden1")
      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden2")
      self.hidden_layer3 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden3")
      self.hidden_layer4 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden4")
      self.hidden_layer5 = tf.keras.layers.Dense(units=size_hidden_layer, activation=tf.nn.relu, name="hidden5")
      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.hidden_layer1(self.input_layer)
      layer = self.hidden_layer2(layer)
      layer = self.hidden_layer3(layer)
      layer = self.hidden_layer4(layer)
      layer = self.hidden_layer5(layer)
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc128]", end=" ")
      print(ex)



  def train_model(self, input, answer, validation):
    try:
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="min")
      hist = self.model.fit(input, answer, epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping])
      #hist = self.model.fit(input, answer, epochs=learning_epoch, validation_data=validation)
      tf.keras.experimental.export_saved_model(self.model, model_full_path)

      file = open(log_full_path, "w")
      file.write("loss\n")
      file.write(str(hist.history['loss']) + "\n\n")
      file.write("val_loss\n")
      file.write(str(hist.history['val_loss']) + "\n\n")
      file.close()

      return hist

    except Exception as ex:
      print("[Autoencoder.train_model]", end=" ")
      print(ex)



  def restore_model(self, path):
    try:
      self.model = tf.keras.experimental.load_from_saved_model(model_path + path)
      self.model.summary()

    except Exception as ex:
      print("[Autoencoder.restore_model]", end=" ")
      print(ex)



  def test_model(self, input):
    try:
      return self.model.predict(input)

    except Exception as ex:
      print("[Autoencoder.test_model]", end=" ")
      print(ex)
