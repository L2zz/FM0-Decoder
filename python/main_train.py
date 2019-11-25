import random
import numpy as np

from Autoencoder import Autoencoder
from ExecutionTime import ExecutionTime
from SignalSet import SignalSet

from read_file import read_file
from make_set import make_set
from test_set import test_set
from plot_log import plot_log

import global_vars
file_name_list = global_vars.file_name_list



if __name__ == "__main__":
  tot_time = ExecutionTime("TOTAL")
  autoencoder = Autoencoder()
  input_set = SignalSet()
  answer_set = SignalSet()



  print("\n\n\n\t\t\t***** READING FILES *****")
  read_time = ExecutionTime("READ")
  for file_name in file_name_list:
    try:
      input, answer = read_file(file_name)
      input, answer = make_set(input, answer)
      input_set.concatenate(input)
      answer_set.concatenate(answer)

    except Exception as ex:
      print("[main_train.py]", end=" ")
      print(ex)

  random_index = [i for i in range(len(input_set.train))]
  random.shuffle(random_index)
  input_set.random_train_set(random_index)
  answer_set.random_train_set(random_index)
  read_time.stop(True)



  print("\n\n\n\t\t\t***** TRAINING *****")
  train_time = ExecutionTime("TRAIN")
  hist = autoencoder.train_model(np.array(input_set.train), np.array(answer_set.train), (np.array(input_set.validation), np.array(answer_set.validation)))
  train_time.stop(True)

  test_time = ExecutionTime("TEST")
  success = test_set(autoencoder.test_model(np.array(input_set.test)), answer_set.test)
  test_time.stop(False)

  plot_log(["model"], [np.array(hist.history['loss'])], [np.array(hist.history['val_loss'])], [float(success) / len(input_set.test)], [True])



  tot_time.stop(False)
  print("\n\n\n\t\t***** SUMMARY *****")
  print("\tTEST RESULT:\t" + str(success) + " / " + str(len(input_set.test)), end=" ")
  print("(" + str(round(100 * (float(success) / len(input_set.test)), 2)) + "%)")
  print("\t\t*****\t*****\t*****")
  tot_time.print()
  print("\t\t*****\t*****\t*****")
  read_time.print()
  train_time.print()
  test_time.print()
  print("\n\n")
