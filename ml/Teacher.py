import datetime
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout

from global_vars import *
from decode_data import decode_enc256


class Teacher(tf.keras.Model):

    def __init__(self):
        try:
            super(Teacher, self).__init__()

            optimizer = tf.keras.optimizers.Adam(lr)
            self.model = self.build_model()
            self.model.compile(loss="mse", optimizer=optimizer,
                               metrics=[decoding_rate])
            self.model.summary()

        except Exception as ex:
            print("[Teacher.__init__]", end=" ")
            print(ex)

    def build_model(self):
        try:
            input_layer = Input(shape=6850, name="teacher_input")

            x = Dense(10000, name="teacher_1")(input_layer)
            x = Activation('relu', name="teacher_activation_1")(x)

            x = Dense(10000, name="teacher_2")(x)
            x = Activation('relu', name="teacher_activation_2")(x)

            x = Dense(5000, name="teacher_3")(x)
            x = Activation('relu', name="teacher_activation_3")(x)

            x = Dense(5000, name="teacher_4")(x)
            x = Activation('relu', name="teacher_activation_4")(x)

            decoded = Dense(268, name="teacher_out")(x)

            return tf.keras.Model(input_layer, decoded)

        except Exception as ex:
            print("[Teacher.build_model]", end=" ")
            print(ex)

    def train_model(self, input, answer, validation):
        try:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_decoding_rate", min_delta=0, patience=10, verbose=1, mode="max")
            best = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path + "_best.h5", monitor='val_decoding_rate',
                                                      verbose=1, save_best_only=True, mode='max')
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
                tf.keras.experimental.export_saved_model(
                    self.model, model_file_path)

            file = open(log_file_path, "w")
            file.write("loss\n")
            file.write(str(hist.history['loss']) + "\n\n")
            file.write("val_loss\n")
            file.write(str(hist.history['val_loss']) + "\n\n")
            file.write("val_decoding_rate\n")
            file.write(str(hist.history['val_decoding_rate']) + "\n\n")
            file.close()

            return hist

        except Exception as ex:
            print("[Teacher.train_model]", end=" ")
            print(ex)

    def test_model(self, input, answer):
        try:
            # self.model = tf.keras.experimental.load_from_saved_model(model_file_path)
            self.model = tf.keras.models.load_model(
                model_file_path + "_best.h5", custom_objects={'decoding_rate': decoding_rate})
            self.model.summary()

            predict = self.model.predict(input)

            # plt.plot(answer[0])
            # plt.plot(predict[0])
            # plt.show()
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


def decoding_rate(y_true, y_pred):

    num = tf.shape(y_pred)[0]
    mask = tf.fill([num], 268)

    zero_arr = tf.fill(tf.shape(y_pred), 0)
    one_arr = tf.fill(tf.shape(y_pred), 1)
    y_pred = tf.where(y_pred < 0.5, zero_arr, one_arr)
    y_pred = tf.dtypes.cast(y_pred, tf.int32)
    y_true = tf.dtypes.cast(y_true, tf.int32)

    tmp = tf.math.equal(y_true, y_pred, name="eq1")
    tmp = tf.math.count_nonzero(tmp, axis=1)
    tmp = tf.dtypes.cast(tmp, tf.int32)
    tmp = tf.math.equal(tmp, mask, name="eq2")
    success = tf.math.count_nonzero(tmp)
    success = tf.dtypes.cast(success, tf.int32)

    return tf.math.divide(success, num)
