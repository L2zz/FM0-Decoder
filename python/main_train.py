import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from global_vars import *
from Network import Network
from ExecutionTime import ExecutionTime
from SignalSet import SignalSet
from read_random_index import read_random_index
from read_file import read_file
from make_set import make_set
from test_set import test_set


if __name__ == "__main__":

    # Set GPU device to run
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # # For auto testing with shell
    try:
        conv2 = int(sys.argv[1])
        conv3 = int(sys.argv[2])
        conv4 = int(sys.argv[3])
        conv5 = int(sys.argv[4])
        conv6 = int(sys.argv[5])

    except Exception as ex:
        conv2 = size_conv_layer2
        conv3 = size_conv_layer3
        conv4 = size_conv_layer4
        conv5 = size_conv_layer5
        conv6 = size_conv_layer6

    try:
        tot_time = ExecutionTime("TOTAL")
        network = Network(conv2, conv3, conv4, conv5, conv6)
        input_set = SignalSet()
        answer_set = SignalSet()
        answer_bit_set = SignalSet()
        index_set = SignalSet()
        read_random_index(index_set)

    except Exception as ex:
        print("[main_train.py: init]", end=" ")
        print(ex)

    print("\n\n\n\t\t\t***** READING FILES *****")
    read_time = ExecutionTime("READ")
    for file_name in file_name_list:
        try:
            input, answer, answer_bit = read_file(file_name)
            input, answer, answer_bit = make_set(input, answer, answer_bit, index_set)
            input_set.concatenate(input)
            answer_set.concatenate(answer)
            answer_bit_set.concatenate(answer_bit)

        except Exception as ex:
            print("[main_train.py: read]", end=" ")
            print(ex)

    try:
        random_index = [i for i in range(len(input_set.train))]
        random.shuffle(random_index)
        input_set.random_train_set(random_index)
        answer_set.random_train_set(random_index)
        read_time.stop(True)

    except Exception as ex:
        print("[main_train.py: shuffle]", end=" ")
        print(ex)

    try:
        print("\n\n\n\t\t\t***** TRAINING *****")
        train_time = ExecutionTime("TRAIN")

        input_set.train = np.array(input_set.train)
        input_set.train = input_set.train.reshape(
            input_set.train.shape[0], 1, input_set.train.shape[1], 1)

        answer_set.train = np.array(answer_set.train)
        answer_set.train = answer_set.train.reshape(
            answer_set.train.shape[0], 1, answer_set.train.shape[1], 1)

        input_set.validation = np.array(input_set.validation)
        input_set.validation = input_set.validation.reshape(
            input_set.validation.shape[0], 1, input_set.validation.shape[1], 1)

        answer_set.validation = np.array(answer_set.validation)
        answer_set.validation = answer_set.validation.reshape(
            answer_set.validation.shape[0], 1, answer_set.validation.shape[1], 1)

        hist = network.train_model(
            input_set.train, answer_set.train, (input_set.validation, answer_set.validation))
        train_time.stop(True)

    except Exception as ex:
        print("[main_train.py: train]", end=" ")
        print(ex)

    try:
        test_time = ExecutionTime("TEST")

        input_set.test = np.array(input_set.test)
        input_set.test = input_set.test.reshape(
            input_set.test.shape[0], 1, input_set.test.shape[1], 1)

        network = Network()
        network.restore_model(execute_time)
        success, success_bit, ber = test_set(
            "SUMMARY", network.test_model(input_set.test), answer_bit_set.test)
        test_time.stop(False)

    except Exception as ex:
        print("[main_train.py: test]", end=" ")
        print(ex)

    try:
        tot_time.stop(False)
        print("\n\n\n\t\t***** SUMMARY *****")
        print("\tTEST RESULT:\t" + str(success) +
              " / " + str(len(input_set.test)), end=" ")
        print("(" + str(round(100 * (float(success) / len(input_set.test)), 2)) + "%)")
        print("\tBER:\t\t" + str(round(100 * ber, 2)) + "%")
        print("\t\t*****\t*****\t*****")
        tot_time.print()
        print("\t\t*****\t*****\t*****")
        read_time.print()
        train_time.print()
        test_time.print()
        print("\n\n")

    except Exception as ex:
        print("[main_train.py: summary]", end=" ")
        print(ex)
