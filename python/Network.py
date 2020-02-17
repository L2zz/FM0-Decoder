import tensorflow as tf
import datetime
from global_vars import *


class Network(tf.keras.Model):

    c1 = 1
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    c7 = 1

    dc1 = 1
    dc2 = 0
    dc3 = 0
    dc4 = 0
    dc5 = 0
    dc6 = 0
    dc7 = 1

    def __init__(self, c2=None, c3=None, c4=None, c5=None, c6=None):
        try:
            super(Network, self).__init__()

            self.c2 = size_conv_layer2 if c2 is None else c2
            self.c3 = size_conv_layer3 if c3 is None else c3
            self.c4 = size_conv_layer4 if c4 is None else c4
            self.c5 = size_conv_layer5 if c5 is None else c5
            self.c6 = size_conv_layer6 if c6 is None else c6

            self.dc2 = self.c2
            self.dc3 = self.c3
            self.dc4 = self.c4
            self.dc5 = self.c5
            self.dc6 = self.c6

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.model = self.build_model()
            self.model.compile(loss="mse", optimizer=optimizer)
            self.model.summary()

        except Exception as ex:
            print("[Network.__init__]", end=" ")
            print(ex)

    def build_model(self):
        try:
            self.input_layer = tf.keras.layers.Input(
                shape=(1, size_input_layer, 1), name="input")

            self.conv_layer1 = tf.keras.layers.Conv2D(
                self.c1, kernel_size=size_kernel, padding=padding, name="conv1")
            self.batch_layer1 = tf.keras.layers.BatchNormalization(
                name="batch1")
            self.activation_layer1 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation1")
            self.dropout_layer1 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout1")

            self.conv_layer2 = tf.keras.layers.Conv2D(
                self.c2, kernel_size=size_kernel, padding=padding, name="conv2")
            self.batch_layer2 = tf.keras.layers.BatchNormalization(
                name="batch2")
            self.activation_layer2 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation2")
            self.dropout_layer2 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout2")

            self.conv_layer3 = tf.keras.layers.Conv2D(
                self.c3, kernel_size=size_kernel, padding=padding, name="conv3")
            self.batch_layer3 = tf.keras.layers.BatchNormalization(
                name="batch3")
            self.activation_layer3 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation3")
            self.dropout_layer3 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout3")

            self.conv_layer4 = tf.keras.layers.Conv2D(
                self.c4, kernel_size=size_kernel, padding=padding, name="conv4")
            self.batch_layer4 = tf.keras.layers.BatchNormalization(
                name="batch4")
            self.activation_layer4 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation4")
            self.dropout_layer4 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout4")

            self.conv_layer5 = tf.keras.layers.Conv2D(
                self.c5, kernel_size=size_kernel, padding=padding, name="conv5")
            self.batch_layer5 = tf.keras.layers.BatchNormalization(
                name="batch5")
            self.activation_layer5 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation5")
            self.dropout_layer5 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout5")

            self.conv_layer6 = tf.keras.layers.Conv2D(
                self.c6, kernel_size=size_kernel, padding=padding, name="conv6")
            self.batch_layer6 = tf.keras.layers.BatchNormalization(
                name="batch6")
            self.activation_layer6 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation6")
            self.dropout_layer6 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout6")

            self.conv_layer7 = tf.keras.layers.Conv2D(
                self.c7, kernel_size=size_kernel, padding=padding, name="conv7")
            self.batch_layer7 = tf.keras.layers.BatchNormalization(
                name="batch7")
            self.activation_layer7 = tf.keras.layers.Activation(
                tf.nn.elu, name="activation7")
            self.dropout_layer7 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout7")

            self.deconv_layer1 = tf.keras.layers.Conv2DTranspose(
                self.dc1, kernel_size=size_kernel, padding=padding, name="deconv1")
            self.deconv_batch_layer1 = tf.keras.layers.BatchNormalization(
                name="deconv_batch1")
            self.deconv_activation_layer1 = tf.keras.layers.Activation(
                tf.nn.elu, name="deconv_activation1")
            self.deconv_dropout_layer1 = tf.keras.layers.Dropout(
                dropout_rate, name="deconv_dropout1")

            self.deconv_layer2 = tf.keras.layers.Conv2DTranspose(
                self.dc2, kernel_size=size_kernel, padding=padding, name="deconv2")
            self.deconv_batch_layer2 = tf.keras.layers.BatchNormalization(
                name="deconv_batch2")
            self.deconv_activation_layer2 = tf.keras.layers.Activation(
                tf.nn.elu, name="deconv_activation2")
            self.deconv_dropout_layer2 = tf.keras.layers.Dropout(
                dropout_rate, name="deconv_dropout2")

            self.deconv_layer3 = tf.keras.layers.Conv2DTranspose(
                self.dc3, kernel_size=size_kernel, padding=padding, name="deconv3")
            self.deconv_batch_layer3 = tf.keras.layers.BatchNormalization(
                name="deconv_batch3")
            self.deconv_activation_layer3 = tf.keras.layers.Activation(
                tf.nn.elu, name="deconv_activation3")
            self.deconv_dropout_layer3 = tf.keras.layers.Dropout(
                dropout_rate, name="deconv_dropout3")

            self.deconv_layer4 = tf.keras.layers.Conv2DTranspose(
                self.dc4, kernel_size=size_kernel, padding=padding, name="deconv4")
            self.deconv_batch_layer4 = tf.keras.layers.BatchNormalization(
                name="deconv_batch4")
            self.deconv_activation_layer4 = tf.keras.layers.Activation(
                tf.nn.elu, name="deconv_activation4")
            self.deconv_dropout_layer4 = tf.keras.layers.Dropout(
                dropout_rate, name="deconv_dropout4")

            self.deconv_layer5 = tf.keras.layers.Conv2DTranspose(
                self.dc5, kernel_size=size_kernel, padding=padding, name="deconv5")
            self.deconv_batch_layer5 = tf.keras.layers.BatchNormalization(
                name="deconv_batch5")
            self.deconv_activation_layer5 = tf.keras.layers.Activation(
                tf.nn.elu, name="deconv_activation5")
            self.deconv_dropout_layer5 = tf.keras.layers.Dropout(
                dropout_rate, name="deconv_dropout5")

            self.deconv_layer6 = tf.keras.layers.Conv2DTranspose(
                self.dc6, kernel_size=size_kernel, padding=padding, name="deconv6")
            self.deconv_batch_layer6 = tf.keras.layers.BatchNormalization(
                name="deconv_batch6")
            self.deconv_activation_layer6 = tf.keras.layers.Activation(
                tf.nn.elu, name="deconv_activation6")
            self.deconv_dropout_layer6 = tf.keras.layers.Dropout(
                dropout_rate, name="deconv_dropout6")

            self.deconv_layer7 = tf.keras.layers.Conv2DTranspose(
                self.dc7, kernel_size=size_kernel, padding=padding, name="deconv7")

            layer = self.dropout_layer1(self.activation_layer1(
                self.batch_layer1(self.conv_layer1(self.input_layer))))
            layer = self.dropout_layer2(self.activation_layer2(
                self.batch_layer2(self.conv_layer2(layer))))
            layer = self.dropout_layer3(self.activation_layer3(
                self.batch_layer3(self.conv_layer3(layer))))
            layer = self.dropout_layer4(self.activation_layer4(
                self.batch_layer4(self.conv_layer4(layer))))
            layer = self.dropout_layer5(self.activation_layer5(
                self.batch_layer5(self.conv_layer5(layer))))
            layer = self.dropout_layer6(self.activation_layer6(
                self.batch_layer6(self.conv_layer6(layer))))
            layer = self.dropout_layer7(self.activation_layer7(
                self.batch_layer7(self.conv_layer7(layer))))

            layer = self.deconv_dropout_layer1(self.deconv_activation_layer1(
                self.deconv_batch_layer1(self.deconv_layer1(layer))))
            layer = self.deconv_dropout_layer2(self.deconv_activation_layer2(
                self.deconv_batch_layer2(self.deconv_layer2(layer))))
            layer = self.deconv_dropout_layer3(self.deconv_activation_layer3(
                self.deconv_batch_layer3(self.deconv_layer3(layer))))
            layer = self.deconv_dropout_layer4(self.deconv_activation_layer4(
                self.deconv_batch_layer4(self.deconv_layer4(layer))))
            layer = self.deconv_dropout_layer5(self.deconv_activation_layer5(
                self.deconv_batch_layer5(self.deconv_layer5(layer))))
            layer = self.deconv_dropout_layer6(self.deconv_activation_layer6(
                self.deconv_batch_layer6(self.deconv_layer6(layer))))
            layer = self.deconv_layer7(layer)

            return tf.keras.Model(self.input_layer, layer)

        except Exception as ex:
            print("[Network.build_model_cnn]", end=" ")
            print(ex)

    def train_model(self, input, answer, validation):
        try:
            best = tf.keras.callbacks.ModelCheckpoint(filepath=model_full_path, monitor='val_loss',
                                                      verbose=1, save_best_only=True)
            if isEarlyStop:
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="min")
                hist = self.model.fit(input, answer, batch_size=batch_size,
                                      epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping, best])
            else:
                hist = self.model.fit(input, answer, batch_size=batch_size,
                                      epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping, best])

            file = open(log_full_path, "w")
            file.write("loss\n")
            file.write(str(hist.history['loss']) + "\n\n")
            file.write("val_loss\n")
            file.write(str(hist.history['val_loss']) + "\n\n")
            file.close()

            return hist

        except Exception as ex:
            print("[Network.train_model]", end=" ")
            print(ex)

    def restore_model(self, name):
        try:
            self.model = tf.keras.models.load_model(model_path + name)
            print("[Loading Success]")

        except Exception as ex:
            print("[Network.restore_model]", end=" ")
            print(ex)
            self.model = tf.keras.experimental.load_from_saved_model(
                model_path + name)
            print("[Loading Success]")

    def test_model(self, input):
        try:
            return self.model.predict(input)

        except Exception as ex:
            print("[Network.test_model]", end=" ")
            print(ex)
