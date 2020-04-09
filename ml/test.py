import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from global_vars import *
from read_file import read_data_file, read_rn_file
from ExecutionTime import ExecutionTime
from Teacher import Teacher
from Student import Student
# from KD import KD


tot_test_file_num = len(file_name_list) * test_file_num

if __name__ == "__main__":

    # Set GPU device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    tot_time = ExecutionTime("TOTAL")
    print("\n\n\n\t\t\t***** READING FILES *****")
    read_time = ExecutionTime("READ")
    rn_file = open(RN_file_path, "r")
    _, _, test_idx = read_rn_file(rn_file)

    is_first = True
    for file_name in file_name_list:
        signal, enc256 = read_data_file(file_name, test_idx)
        if is_first:
            signal_test = signal
            enc256_test = enc256
            is_first = False
        else:
            signal_test += signal
            enc256_test += enc256
    read_time.stop(True)

    try:
        print("\n\n\n\t\t\t***** TESTING *****")
        test_time = ExecutionTime("TEST")
        # loaded = Teacher()
        loaded = Student()
        success, success_bit, ber = loaded.test_model(
            np.array(signal_test), np.array(enc256_test))
        test_time.stop(True)

    except Exception as ex:
        print("[test.py: test]", end=" ")
        print(ex)

    try:
        tot_time.stop(False)
        print("\n\n\n\t\t***** SUMMARY *****")
        print("\tTEST RESULT:\t" + str(success) +
              " / " + str(len(signal_test)), end=" ")
        print("(" + str(round(100 * (float(success) / len(signal_test)), 2)) + "%)")
        print("\tBER:\t\t" + str(round(100 * ber, 2)) + "%")
        print("\t\t*****\t*****\t*****")
        tot_time.print()
        print("\t\t*****\t*****\t*****")
        read_time.print()
        test_time.print()
        print("\n\n")

    except Exception as ex:
        print("[test.py: summary]", end=" ")
        print(ex)
