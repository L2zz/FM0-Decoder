import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

from global_vars import *
from decode_data import decode_enc256


class Teacher(tf.keras.Model):

    def __init__(self):
        try:
            super(Teacher, self).__init__()

            optimizer = tf.keras.optimizers.Adam(lr)
            self.model = self.build_model()
            self.model.compile(loss="mse", optimizer=optimizer)
            self.model.summary()

        except Exception as ex:
            print("[Teacher.__init__]", end=" ")
            print(ex)

    def build_model(self):
        try:
            input_layer = Input(shape=6850, name="teacher_input")

            x = Dense(3350, name="teacher_1")(input_layer)
            x = BatchNormalization(name="teacher_batch_1")(x)
            x = Activation('relu', name="teacher_activation_1")(x)

            x = Dense(3350, name="teacher_2")(x)
            x = BatchNormalization(name="teacher_batch_2")(x)
            x = Activation('relu', name="teacher_activation_2")(x)

            x = Dense(6700, name="teacher_3")(x)
            x = BatchNormalization(name="teacher_batch_3")(x)
            x = Activation('relu', name="teacher_activation_3")(x)

            x = Dense(6700, name="teacher_4")(x)
            x = BatchNormalization(name="teacher_batch_4")(x)
            x = Activation('relu', name="teacher_activation_4")(x)

            x = Dense(1340, name="teacher_5")(x)
            x = BatchNormalization(name="teacher_batch_5")(x)
            x = Activation('relu', name="teacher_activation_5")(x)

            x = Dense(1340, name="teacher_6")(x)
            x = BatchNormalization(name="teacher_batch_6")(x)
            x = Activation('relu', name="teacher_activation_6")(x)

            decoded = Dense(268, name="teacher_7")(x)

            return tf.keras.Model(input_layer, decoded)

        except Exception as ex:
            print("[Teacher.build_model]", end=" ")
            print(ex)

    def train_model(self, input, answer, validation):
        try:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0, patience=5, verbose=1, mode="min")
            best = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path, monitor='val_loss',
                                                      verbose=1, save_best_only=True)
            if isEarlyStop:
                if isBestSave:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=epochs, validation_data=validation, callbacks=[early_stopping, best])
                else:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=epochs, validation_data=validation, callbacks=[early_stopping])
            else:
                if isBestSave:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=epochs, validation_data=validation, callbacks=[best])
                else:
                    hist = self.model.fit(input, answer, batch_size=batch_size,
                                          epochs=epochs, validation_data=validation, callbacks=[])

            if not isBestSave:
                tf.keras.experimental.export_saved_model(self.model, model_file_path)

            file = open(log_file_path, "w")
            file.write("loss\n")
            file.write(str(hist.history['loss']) + "\n\n")
            file.write("val_loss\n")
            file.write(str(hist.history['val_loss']) + "\n\n")
            file.close()

            return hist

        except Exception as ex:
            print("[Teacher.train_model]", end=" ")
            print(ex)

    def test_model(self, input, answer):
        try:
            self.model = tf.keras.experimental.load_from_saved_model(model_file_path)
            self.model.summary()

            predict = self.model.predict(input)
            plt.plot(answer[0])
            plt.plot(predict[0])
            plt.show()
            # for i in range(len(input)):
            #     if i % 100 == 0:
            #         plt.plot(answer[i])
            #         plt.plot(predict[i])
            #         plt.show()
            success, success_bit, ber = decode_enc256(predict, answer)

            return success, success_bit, ber

        except Exception as ex:
            print("[Teacher.test_model]", end=" ")
            print(ex)
