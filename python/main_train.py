import random
import numpy as np

from global_vars import *
from Autoencoder import Autoencoder
from ExecutionTime import ExecutionTime
from SignalSet import SignalSet

from read_random_index import read_random_index
from read_file import read_file
from make_set import make_set
from test_set import test_set



if __name__ == "__main__":
  try:
    tot_time = ExecutionTime("TOTAL")
    autoencoder = Autoencoder()
    input_set = SignalSet()
    answer_set = SignalSet()
    index_set = SignalSet()
    read_random_index(index_set)

  except Exception as ex:
    print("[main_train.py: init]", end=" ")
    print(ex)



  print("\n\n\n\t\t\t***** READING FILES *****")
  read_time = ExecutionTime("READ")
  for file_name in file_name_list:
    try:
      input, answer = read_file(file_name)
      input, answer = make_set(input, answer, index_set)
      input_set.concatenate(input)
      answer_set.concatenate(answer)

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
    hist = autoencoder.train_model(np.array(input_set.train), np.array(answer_set.train), (np.array(input_set.validation), np.array(answer_set.validation)))
    train_time.stop(True)

  except Exception as ex:
    print("[main_train.py: train]", end=" ")
    print(ex)



  try:
    test_time = ExecutionTime("TEST")
    success, success_bit, ber = test_set("SUMMARY", autoencoder.test_model(np.array(input_set.test)), answer_set.test)
    test_time.stop(False)

  except Exception as ex:
    print("[main_train.py: test]", end=" ")
    print(ex)



  try:
    tot_time.stop(False)
    print("\n\n\n\t\t***** SUMMARY *****")
    print("\tTEST RESULT:\t" + str(success) + " / " + str(len(input_set.test)), end=" ")
    print("(" + str(round(100 * (float(success) / len(input_set.test)), 2)) + "%)")
    print("\tBER:\t\t" + str(round(100*ber, 2)) + "%")
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
