import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from global_vars import *
from read_file import read_data_file, read_rn_file
from ExecutionTime import ExecutionTime
from Teacher import Teacher
from Student import Student
from KD import KD


if __name__ == "__main__":

    # Set GPU device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    tot_time = ExecutionTime("TOTAL")
    print("\n\n\n\t\t\t***** READING FILES *****")
    read_time = ExecutionTime("READ")
    rn_file = open(RN_file_path, "r")
    train_idx, valid_idx, _ = read_rn_file(rn_file)

    is_first = True
    for file_name in file_name_list:
        signal, enc256 = read_data_file(file_name, train_idx)
        if is_first:
            signal_train = signal
            enc256_train = enc256
        else:
            signal_train += signal
            enc256_train += enc256

        signal, enc256 = read_data_file(file_name, valid_idx)
        if is_first:
            signal_valid = signal
            enc256_valid = enc256
            is_first = False
        else:
            signal_valid += signal
            enc256_valid += enc256
    read_time.stop(True)

    try:
        print("\n\n\n\t\t\t***** TRAINING *****")
        train_time = ExecutionTime("TRAIN")
        model = Teacher()
        # model = Student()
        model.train_model(np.array(signal_train), np.array(enc256_train),
                          (np.array(signal_valid), np.array(enc256_valid)))

        # teacher = tf.keras.experimental.load_from_saved_model(teacher_file_path)
        # tpredict_train = teacher.predict(np.array(signal_train))
        # tpredict_valid = teacher.predict(np.array(signal_valid))
        # model = KD()
        # model.train_model(np.array(signal_train), [np.array(enc256_train), tpredict_train],
        #                   (np.array(signal_valid), [np.array(enc256_valid), tpredict_valid]))
        train_time.stop(True)

    except Exception as ex:
        print("[main_train.py: train]", end=" ")
        print(ex)

    tot_time.stop(False)
    tot_time.print()
