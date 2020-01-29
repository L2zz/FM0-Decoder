import tensorflow as tf
import datetime
from global_vars import *


class Network(tf.keras.Model):

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0
    c7 = 0
    c8 = 0
    c9 = 0
    c10 = 0
    fc1 = 0
    fc2 = 0
    fc3 = 0

    def __init__(self):
        try:
            super(Network, self).__init__()

            self.c1 = size_conv_layer1
            self.c2 = size_conv_layer2
            self.c3 = size_conv_layer3
            self.c4 = size_conv_layer4
            self.c5 = size_conv_layer5
            self.c6 = size_conv_layer6
            self.c7 = size_conv_layer7
            self.c8 = size_conv_layer8
            self.c9 = size_conv_layer9
            self.c10 = size_conv_layer10
            self.fc1 = size_fc_layer1
            self.fc2 = size_fc_layer2
            self.fc3 = size_fc_layer3

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.model = eval("self.build_model_cnn" + type)()
            self.model.compile(loss="mse", optimizer=optimizer)
            self.model.summary()

        except Exception as ex:
            print("[Network.__init__]", end=" ")
            print(ex)

    def __init__(self, c1, c2, c3, c4, fc):
        try:
            super(Network, self).__init__()

            self.c1 = c1
            self.c2 = c1
            self.c3 = c2
            self.c4 = c2
            self.c5 = c3
            self.c6 = c3
            self.c7 = c3
            self.c8 = c4
            self.c9 = c4
            self.c10 = c4
            self.fc1 = fc
            self.fc2 = fc
            self.fc3 = fc

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.model = eval("self.build_model_cnn" + type)()
            self.model.compile(loss="mse", optimizer=optimizer)
            self.model.summary()

        except Exception as ex:
            print("[Network.__init__]", end=" ")
            print(ex)

    def build_model_cnn_vgg(self):
        try:
            self.input_layer = tf.keras.layers.Input(
                shape=(size_input_layer, 1), name="input")

            self.conv_layer1 = tf.keras.layers.Conv1D(
                self.c1, kernel_size=size_kernel, padding=padding, name="conv1")
            self.batch_layer1 = tf.keras.layers.BatchNormalization(
                name="batch1")
            self.activation_layer1 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation1")
            self.dropout_layer1 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout1")
            self.conv_layer2 = tf.keras.layers.Conv1D(
                self.c2, kernel_size=size_kernel, padding=padding, name="conv2")
            self.batch_layer2 = tf.keras.layers.BatchNormalization(
                name="batch2")
            self.activation_layer2 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation2")
            self.dropout_layer2 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout2")
            self.pooling_layer1 = tf.keras.layers.MaxPooling1D(size_pool)

            self.conv_layer3 = tf.keras.layers.Conv1D(
                self.c3, kernel_size=size_kernel, padding=padding, name="conv3")
            self.batch_layer3 = tf.keras.layers.BatchNormalization(
                name="batch3")
            self.activation_layer3 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation3")
            self.dropout_layer3 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout3")
            self.conv_layer4 = tf.keras.layers.Conv1D(
                self.c4, kernel_size=size_kernel, padding=padding, name="conv4")
            self.batch_layer4 = tf.keras.layers.BatchNormalization(
                name="batch4")
            self.activation_layer4 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation4")
            self.dropout_layer4 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout4")
            self.pooling_layer2 = tf.keras.layers.MaxPooling1D(size_pool)

            self.conv_layer5 = tf.keras.layers.Conv1D(
                self.c5, kernel_size=size_kernel, padding=padding, name="conv5")
            self.batch_layer5 = tf.keras.layers.BatchNormalization(
                name="batch5")
            self.activation_layer5 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation5")
            self.dropout_layer5 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout5")
            self.conv_layer6 = tf.keras.layers.Conv1D(
                self.c6, kernel_size=size_kernel, padding=padding, name="conv6")
            self.batch_layer6 = tf.keras.layers.BatchNormalization(
                name="batch6")
            self.activation_layer6 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation6")
            self.dropout_layer6 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout6")
            self.pooling_layer3 = tf.keras.layers.MaxPooling1D(size_pool)

            self.conv_layer7 = tf.keras.layers.Conv1D(
                self.c7, kernel_size=size_kernel, padding=padding, name="conv7")
            self.batch_layer7 = tf.keras.layers.BatchNormalization(
                name="batch7")
            self.activation_layer7 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation7")
            self.dropout_layer7 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout7")
            self.conv_layer8 = tf.keras.layers.Conv1D(
                self.c8, kernel_size=size_kernel, padding=padding, name="conv8")
            self.batch_layer8 = tf.keras.layers.BatchNormalization(
                name="batch8")
            self.activation_layer8 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation8")
            self.dropout_layer8 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout8")
            self.pooling_layer4 = tf.keras.layers.MaxPooling1D(size_pool)


            self.conv_layer9 = tf.keras.layers.Conv1D(
                self.c9, kernel_size=size_kernel, padding=padding, name="conv9")
            self.batch_layer9 = tf.keras.layers.BatchNormalization(
                name="batch9")
            self.activation_layer9 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation9")
            self.dropout_layer9 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout9")
            self.conv_layer10 = tf.keras.layers.Conv1D(
                self.c10, kernel_size=size_kernel, padding=padding, name="conv10")
            self.batch_layer10 = tf.keras.layers.BatchNormalization(
                name="batch10")
            self.activation_layer10 = tf.keras.layers.Activation(
                tf.nn.relu, name="activation10")
            self.dropout_layer10 = tf.keras.layers.Dropout(
                dropout_rate, name="dropout10")
            self.pooling_layer5 = tf.keras.layers.MaxPooling1D(size_pool)

            self.flatten_layer = tf.keras.layers.Flatten()

            self.fc_layer1 = tf.keras.layers.Dense(
                units=self.fc1, name="fc1")
            self.batch_layerf1 = tf.keras.layers.BatchNormalization(
                name="batchf1")
            self.activation_layerf1 = tf.keras.layers.Activation(
                tf.nn.relu, name="activationf1")
            self.dropout_layerf1 = tf.keras.layers.Dropout(
                dropout_rate, name="dropoutf1")

            self.fc_layer2 = tf.keras.layers.Dense(
                units=self.fc2, name="fc2")
            self.batch_layerf2 = tf.keras.layers.BatchNormalization(
                name="batchf2")
            self.activation_layerf2 = tf.keras.layers.Activation(
                tf.nn.relu, name="activationf2")
            self.dropout_layerf2 = tf.keras.layers.Dropout(
                dropout_rate, name="dropoutf2")

            self.output_layer = tf.keras.layers.Dense(
                units=size_output_layer, name="output")

            layer = self.dropout_layer1(self.activation_layer1(
                self.batch_layer1(self.conv_layer1(self.input_layer))))
            layer = self.dropout_layer2(self.activation_layer2(
                self.batch_layer2(self.conv_layer2(layer))))
            layer = self.pooling_layer1(layer)

            layer = self.dropout_layer3(self.activation_layer3(
                self.batch_layer3(self.conv_layer3(layer))))
            layer = self.dropout_layer4(self.activation_layer4(
                self.batch_layer4(self.conv_layer4(layer))))
            layer = self.pooling_layer2(layer)

            layer = self.dropout_layer5(self.activation_layer5(
                self.batch_layer5(self.conv_layer5(layer))))
            layer = self.dropout_layer6(self.activation_layer6(
                self.batch_layer6(self.conv_layer6(layer))))
            layer = self.pooling_layer3(layer)

            layer = self.dropout_layer7(self.activation_layer7(
                self.batch_layer7(self.conv_layer7(layer))))
            layer = self.dropout_layer8(self.activation_layer8(
                self.batch_layer8(self.conv_layer8(layer))))
            layer = self.pooling_layer4(layer)

            layer = self.dropout_layer9(self.activation_layer9(
                self.batch_layer9(self.conv_layer9(layer))))
            layer = self.dropout_layer10(self.activation_layer10(
                self.batch_layer10(self.conv_layer10(layer))))
            layer = self.pooling_layer5(layer)

            layer = self.flatten_layer(layer)
            layer = self.dropout_layerf1(self.activation_layerf1(
                self.batch_layerf1(self.fc_layer1(layer))))
            layer = self.dropout_layerf2(self.activation_layerf2(
                self.batch_layerf2(self.fc_layer2(layer))))
            layer = self.output_layer(layer)

            return tf.keras.Model(self.input_layer, layer)

        except Exception as ex:
            print("[Network.build_model_cnn]", end=" ")
            print(ex)

    def train_model(self, input, answer, validation):
        try:
            if isEarlyStop:
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="min")
                hist = self.model.fit(input, answer, batch_size=batch_size,
                                      epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping])
            else:
                hist = self.model.fit(input, answer, batch_size=batch_size,
                                      epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping])
            tf.keras.experimental.export_saved_model(
                self.model, model_full_path)

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

    def restore_model(self, path):
        try:
            self.model = tf.keras.experimental.load_from_saved_model(
                model_path + path)
            self.model.summary()

        except Exception as ex:
            print("[Network.restore_model]", end=" ")
            print(ex)

    def test_model(self, input):
        try:
            return self.model.predict(input)

        except Exception as ex:
            print("[Network.test_model]", end=" ")
            print(ex)
