import numpy as np
from tqdm import tqdm

from ExecutionTime import ExecutionTime

from read_file import read_file
from standardize_sample import standardize_sample
from decode_data import decode_data

import global_vars
log_path = global_vars.log_path
bit_data = global_vars.bit_data
file_size = global_vars.file_size
file_name_list = global_vars.file_name_list
tot_file_size = len(file_name_list) * file_size



if __name__ == "__main__":
  tot_time = ExecutionTime("TOTAL")
  file = open(log_path, "w")
  success_bit = np.zeros(bit_data + 1)
  success = 0



  for file_name in file_name_list:
    try:
      file_time = ExecutionTime(file_name)
      print("\n\n\t*** " + file_name + " ***")
      file.write("\t*** " + file_name + " ***\n")
      signal, databit = read_file(file_name)
      signal = standardize_sample(signal)



      cur_success_bit = np.zeros(bit_data + 1)
      cur_success = 0
      for idx in tqdm(range(file_size), desc="DECODING", ncols=100, unit=" signal"):
        decoded_bit = decode_data(signal[idx])



        count = 0
        for bit_idx in range(bit_data):
          if decoded_bit[bit_idx] == databit[idx][bit_idx]: count += 1
        if count == bit_data:
          cur_success += 1
          success += 1
        cur_success_bit[count] += 1
        success_bit[count] += 1



      print("\n\tRESULT: " + str(cur_success) + " / " + str(file_size), end=" ")
      print("(" + str(round(100 * (float(cur_success) / file_size), 2)) + "%)")
      file_time.stop(False)

      file.write("RESULT: " + str(cur_success) + " / " + str(file_size))
      file.write("(" + str(round(100 * (float(cur_success) / file_size), 2)) + "%)\n")
      file_time.print_file(file)
      for data in cur_success_bit:
        file.write(str(int(data)) + " ")
      file.write("\n\n\n")



    except Exception as ex:
      print("[main.py]", end=" ")
      print(ex)
      file_time.stop(False)



  print("\n\n\t*** SUMMARY ***")
  print("\tTOTALRESULT: " + str(success) + " / " + str(tot_file_size), end=" ")
  print("(" + str(round(100 * (float(success) / tot_file_size), 2)) + "%)")
  tot_time.stop(True)
  print("\n")

  file.write("\t*** SUMMARY ***\n")
  file.write("RESULT: " + str(success) + " / " + str(tot_file_size))
  file.write("(" + str(round(100 * (float(success) / tot_file_size), 2)) + "%)\n")
  tot_time.print_file(file)
  for data in success_bit:
    file.write(str(int(data)) + " ")

  file.close()
