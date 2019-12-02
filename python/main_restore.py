import os
import numpy as np

from global_vars import *
from Autoencoder import Autoencoder
from ExecutionTime import ExecutionTime
from SignalSet import SignalSet

from read_testSet import read_testSet
from test_set import test_set



if __name__ == "__main__":
  model_name = ""

  try:
    print("\n\n\n" + str(os.listdir(model_path)))
    model_name = input("\n\n\nInput the model name you want to restore: ").rstrip("\n")
    if not os.path.exists(model_path + model_name):
      raise NameError("Model \"" + model_name + "\" does not exist")

  except Exception as ex:
    print("[main_restore.py: select_model]", end=" ")
    print(ex)



  try:
    tot_time = ExecutionTime("TOTAL")
    autoencoder = Autoencoder()

    print("\n\n\n\t\t\t***** RESTORE MODEL *****")
    restore_time = ExecutionTime("RESTORE")
    autoencoder.restore_model(model_name)
    restore_time.stop(False)

  except Exception as ex:
    print("[main_restore.py: restore]", end=" ")
    print(ex)


  try:
    print("\n\n\n\t\t\t***** TEST MODEL *****")
    test_time = ExecutionTime("TEST")
    tot_success = 0
    tot_success_bit = np.zeros(257)
    tot_file = 0

    for file_name in file_name_list:
      try:
        input, answer = read_testSet(file_name)
        success, success_bit, ber = test_set(file_name, autoencoder.test_model(np.array(input)), answer)

        print("\n\tTEST RESULT:\t" + str(success) + " / " + str(len(input)), end=" ")
        print("(" + str(round(100 * (float(success) / len(input)), 2)) + "%)")
        print("\tBER:\t\t" + str(round(100*ber, 2)) + "%\n\n")
        tot_success += success
        tot_success_bit = np.add(tot_success_bit, success_bit)
        tot_file += 1

      except Exception as ex:
        print("[main_restore: " + file_name + "]", end=" ")
        print(ex)

    test_time.stop(False)

  except Exception as ex:
    print("[main_restore: test]", end=" ")
    print(ex)



  try:
    ber = 0
    for idx in range(257):
      ber += idx * tot_success_bit[idx]
    ber = 1 - ber / (tot_file*test_size*256)
    tot_time.stop(False)

    print("\n\n\n\t\t***** SUMMARY *****")
    print("\tTEST RESULT:\t" + str(tot_success) + " / " + str(tot_file*test_size), end=" ")
    print("(" + str(round(100 * (float(tot_success) / (tot_file*test_size)), 2)) + "%)")
    print("\tBER:\t\t" + str(round(100*ber, 2)) + "%")
    print("\t\t*****\t*****\t*****")
    tot_time.print()
    print("\t\t*****\t*****\t*****")
    restore_time.print()
    test_time.print()
    print("\n\n")

    file = open(log_full_path, "a")
    file.write("\t\t\t***** SUMMARY *****\n\n")
    file.write(str(tot_success) + " / " + str(tot_file*test_size))
    file.write(" (" + str(round(100 * (float(tot_success) / (tot_file*test_size)), 2)) + "%)\n\n")
    for idx in range(257):
      file.write(str(int(tot_success_bit[idx])) + " ")
    file.write("\n\nBER: " + str(ber))
    file.close()

  except Exception as ex:
    print("[main_restore.py: summary]", end=" ")
    print(ex)
