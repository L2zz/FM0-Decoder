import os
import numpy as np

from Autoencoder import Autoencoder
from ExecutionTime import ExecutionTime
from SignalSet import SignalSet

from read_file import read_file
from test_set import test_set

import global_vars
file_name_list = global_vars.file_name_list
model_path = global_vars.model_path




if __name__ == "__main__":
  model_name = ""

  try:
    print("\n\n\n" + str(os.listdir(model_path)))
    model_name = input("\n\n\nInput the model name you want to restore: ").rstrip("\n")
    if not os.path.exists(model_path + model_name):
      raise NameError("Model \"" + model_name + "\" does not exist")

  except Exception as ex:
    print("[main_restore.py]", end=" ")
    print(ex)



  tot_time = ExecutionTime("TOTAL")
  autoencoder = Autoencoder()
  input_set = []
  answer_set = []



  print("\n\n\n\t\t\t***** READING FILES *****")
  read_time = ExecutionTime("READ")
  for file_name in file_name_list:
    try:
      input, answer = read_file(file_name)
      input_set += input
      answer_set += answer

    except Exception as ex:
      print("[main_restore.py]", end=" ")
      print(ex)
  read_time.stop(True)



  print("\n\n\n\t\t\t***** RESTORE MODEL *****")
  restore_time = ExecutionTime("RESTORE")
  autoencoder.restore_model(model_name)
  restore_time.stop(False)



  print("\n\n\n\t\t\t***** TEST MODEL *****")
  test_time = ExecutionTime("TEST")
  success = test_set(autoencoder.test_model(np.array(input_set)), answer_set)
  test_time.stop(False)



  tot_time.stop(False)
  print("\n\n\n\t\t***** SUMMARY *****")
  print("\tTEST RESULT:\t" + str(success) + " / " + str(len(input_set)), end=" ")
  print("(" + str(round(100 * (float(success) / len(input_set)), 2)) + "%)")
  print("\t\t*****\t*****\t*****")
  tot_time.print()
  print("\t\t*****\t*****\t*****")
  read_time.print()
  restore_time.print()
  test_time.print()
  print("\n\n")
