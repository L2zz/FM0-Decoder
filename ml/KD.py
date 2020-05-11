import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation

from global_vars import *
from decode_data import decode_enc256


class KD(tf.keras.Model):

    def __init__(self):
        try:
            super(KD, self).__init__()

            optimizer = tf.keras.optimizers.Adam(lr)
            self.model = self.build_model()
            self.model.compile(
                loss="mse", loss_weights=[alpha, (1 - alpha)],
                optimizer=optimizer, metrics={'kd_out': decoding_rate})
            self.model.summary()

        except Exception as ex:
            print("[KD.__init__]", end=" ")
            print(ex)

    def build_model(self):
        try:
            input_layer = Input(shape=6850, name="kd_input")

            x = Dense(2000, name="kd_1")(input_layer)
            x = Activation('relu', name="kd_activation_1")(x)

            x = Dense(2000, name="kd_2")(x)
            x = Activation('relu', name="kd_activation_2")(x)

            x = Dense(1000, name="kd_3")(x)
            x = Activation('relu', name="kd_activation_3")(x)

            # x = Dense(1000, name="kd_4")(x)
            # x = Activation('relu', name="kd_activation_4")(x)

            student_output = Dense(268, name="kd_out")(x)
            self.student = tf.keras.Model(input_layer, student_output)

            outputs = [student_output, student_output]

            return tf.keras.Model(inputs=input_layer, outputs=outputs)

        except Exception as ex:
            print("[KD.build_model]", end=" ")
            print(ex)

    def train_model(self, input, answer, validation):
        try:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_kd_out_decoding_rate', min_delta=0, patience=10, verbose=1, mode="max")
            best = tf.keras.callbacks.ModelCheckpoint(filepath=model_file_path + "_best.h5", monitor='val_kd_out_decoding_rate',
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
                    self.student, model_file_path)

            file = open(log_file_path, "w")
            file.write("loss\n")
            file.write(str(hist.history['loss']) + "\n\n")
            file.write("val_loss\n")
            file.write(str(hist.history['val_loss']) + "\n\n")
            file.write("val_decoding_rate\n")
            file.write(str(hist.history['val_kd_out_decoding_rate']) + "\n\n")
            file.close()

            return hist

        except Exception as ex:
            print("[KD.train_model]", end=" ")
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
