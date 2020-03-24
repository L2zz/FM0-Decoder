import datetime

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D

from global_vars import *


class Network(tf.keras.Model):

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0
    c6 = 0

    dc1 = 0
    dc2 = 0
    dc3 = 0
    dc4 = 0
    dc5 = 0
    dc6 = 0

    dc7 = 2

    def __init__(self, c1=size_conv_layer1, c2=size_conv_layer3, c3=size_conv_layer5):
        try:
            super(Network, self).__init__()

            self.c1 = c1
            self.c2 = self.c1
            self.c3 = c2
            self.c4 = self.c3
            self.c5 = c3
            self.c6 = self.c5

            self.dc1 = self.c6
            self.dc2 = self.c5
            self.dc3 = self.c4
            self.dc4 = self.c3
            self.dc5 = self.c2
            self.dc6 = self.c1

            optimizer = tf.keras.optimizers.Adam(learning_rate)
            self.model = self.build_model()
            self.model.compile(loss="mse", optimizer=optimizer)
            self.model.summary()

        except Exception as ex:
            print("[Network.__init__]", end=" ")
            print(ex)

    def build_model(self):
        try:
            input_sig = Input(shape=(1, 6848, 1))

            x = Conv2D(256, (1, 3), activation='relu', padding='same')(input_sig)
            x = MaxPooling2D((1, 2))(x)
            x = Conv2D(128, (1, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((1, 2))(x)
            x = Conv2D(128, (1, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((1, 2))(x)
            x = Conv2D(64, (1, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((1, 2))(x)
            x = Conv2D(64, (1, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((1, 2))(x)
            x = Conv2D(32, (1, 3), activation='relu', padding='same')(x)
            x = MaxPooling2D((1, 2))(x)
            x = Conv2D(32, (1, 3), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((1, 2))(x)

            x = Conv2D(32, (1, 3), activation='relu', padding='same')(encoded)
            x = UpSampling2D((1, 2))(x)
            x = ZeroPadding2D(padding=((0, 0), (0, 1)))(x)
            x = Conv2D(32, (1, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((1, 2))(x)
            x = Conv2D(64, (1, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((1, 2))(x)
            x = Conv2D(64, (1, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((1, 2))(x)
            x = Conv2D(128, (1, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((1, 2))(x)
            x = Conv2D(128, (1, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((1, 2))(x)
            x = Conv2D(256, (1, 3), activation='relu', padding='same')(x)
            x = UpSampling2D((1, 2))(x)
            decoded = Conv2D(1, (1, 3), activation='sigmoid', padding='same')(x)

            return tf.keras.Model(input_sig, decoded)

        except Exception as ex:
            print("[Network.build_model_cnn]", end=" ")
            print(ex)

    def train_model(self, input, answer, validation):
        try:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="min")
            best = tf.keras.callbacks.ModelCheckpoint(filepath=model_full_path, monitor='val_loss',
                                                      verbose=1, save_best_only=True)
            if isEarlyStop:
                if isBestSave:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping, best])
                else:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=learning_epoch, validation_data=validation, callbacks=[early_stopping])
            else:
                if isBestSave:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=learning_epoch, validation_data=validation, callbacks=[best])
                else:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=learning_epoch, validation_data=validation, callbacks=[])

            if not isBestSave:
                tf.keras.experimental.export_saved_model(self.model, model_full_path)

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

    def test_model(self, input, answer):
        try:
            predict = self.model.predict(input)
            # plt.plot(answer[0, 0, :, 0])
            # plt.plot(predict[0, 0, :, 0])
            # plt.show()
            for i in range(len(input)):
                if i % 100 == 0:
                    plt.plot(answer[i, 0, :, 0])
                    plt.plot(predict[i, 0, :, 0])
                    plt.show()

            loss = self.model.evaluate(input, answer)
            print("\n\t\t***** LOSS *****")
            print("\tLOSS: " + str(loss))
            print("\n\t\t****************")

            return predict

        except Exception as ex:
            print("[Network.test_model]", end=" ")
            print(ex)
