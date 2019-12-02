import tensorflow as tf
import datetime
from global_vars import *



class Autoencoder(tf.keras.Model):
  def __init__(self):
    try:
      super(Autoencoder, self).__init__()

      optimizer = tf.keras.optimizers.Adam(learning_rate)
      self.model = eval("self.build_model" + tail)()
      self.model.compile(loss="mse", optimizer=optimizer)
      self.model.summary()

    except Exception as ex:
      print("[Autoencoder.__init__]", end=" ")
      print(ex)



  def build_model_enc256(self):
    try:
      self.input_layer = tf.keras.layers.Input(shape=(size_input_layer,), name="input")

      self.hidden_layer1 = tf.keras.layers.Dense(units=size_hidden_layer, name="hidden1")
      self.batch_layer1 = tf.keras.layers.BatchNormalization(name="batch1")
      self.activation_layer1 = tf.keras.layers.Activation(tf.nn.relu, name="activation1")
      self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate, name="dropout1")

      self.hidden_layer2 = tf.keras.layers.Dense(units=size_hidden_layer2, name="hidden2")
      self.batch_layer2 = tf.keras.layers.BatchNormalization(name="batch2")
      self.activation_layer2 = tf.keras.layers.Activation(tf.nn.relu, name="activation2")
      self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate, name="dropout2")

      self.hidden_layer3 = tf.keras.layers.Dense(units=size_hidden_layer3, name="hidden3")
      self.batch_layer3 = tf.keras.layers.BatchNormalization(name="batch3")
      self.activation_layer3 = tf.keras.layers.Activation(tf.nn.relu, name="activation3")
      self.dropout_layer3 = tf.keras.layers.Dropout(dropout_rate, name="dropout3")

      self.hidden_layer4 = tf.keras.layers.Dense(units=size_hidden_layer4, name="hidden4")
      self.batch_layer4 = tf.keras.layers.BatchNormalization(name="batch4")
      self.activation_layer4 = tf.keras.layers.Activation(tf.nn.relu, name="activation4")
      self.dropout_layer4 = tf.keras.layers.Dropout(dropout_rate, name="dropout4")

      self.hidden_layer5 = tf.keras.layers.Dense(units=size_hidden_layer5, name="hidden5")
      self.batch_layer5 = tf.keras.layers.BatchNormalization(name="batch5")
      self.activation_layer5 = tf.keras.layers.Activation(tf.nn.relu, name="activation5")
      self.dropout_layer5 = tf.keras.layers.Dropout(dropout_rate, name="dropout5")

      self.hidden_layer6 = tf.keras.layers.Dense(units=size_hidden_layer6, name="hidden6")
      self.batch_layer6 = tf.keras.layers.BatchNormalization(name="batch6")
      self.activation_layer6 = tf.keras.layers.Activation(tf.nn.relu, name="activation6")
      self.dropout_layer6 = tf.keras.layers.Dropout(dropout_rate, name="dropout6")

      self.output_layer = tf.keras.layers.Dense(units=size_output_layer, name="output")

      layer = self.dropout_layer1(self.activation_layer1(self.batch_layer1(self.hidden_layer1(self.input_layer))))
      layer = self.dropout_layer2(self.activation_layer2(self.batch_layer2(self.hidden_layer2(layer))))
      layer = self.dropout_layer3(self.activation_layer3(self.batch_layer3(self.hidden_layer3(layer))))
      layer = self.dropout_layer4(self.activation_layer4(self.batch_layer4(self.hidden_layer4(layer))))
      layer = self.dropout_layer5(self.activation_layer5(self.batch_layer5(self.hidden_layer5(layer))))
      if sample_type == "org":
        layer = self.dropout_layer6(self.activation_layer6(self.batch_layer6(self.hidden_layer6(layer))))
      layer = self.output_layer(layer)
      return tf.keras.Model(self.input_layer, layer)

    except Exception as ex:
      print("[Autoencoder.build_model_enc256]", end=" ")
      print(ex)



  def train_model(self, input, answer, validation):
    try:
      if isEarlyStop:
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="min")
        hist = self.model.fit(input, answer, epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping])
      else:
        hist = self.model.fit(input, answer, epochs=learning_epoch, validation_data=validation)
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
