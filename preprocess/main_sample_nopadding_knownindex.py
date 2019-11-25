from tqdm import tqdm

from read_file import read_file
from standardize_sample import standardize_sample
from detect_preamble import detect_preamble

from global_vars import *
num_bit_data = bit_data * 2 * num_half_bit



if __name__ == "__main__":
  fileP = open(data_path + "index_std_cut", "r")
  line = fileP.readline()

  for file_name in file_name_list:
    try:
      print("\n\n\t*** " + file_name + " ***")
      signal = read_file(file_name)
      line = fileP.readline()
      line = fileP.readline()
      start = fileP.readline().rstrip(" \n").split(" ")
      start = [int(i) for i in start]

      file = open(target_path + file_name + "_sample", "w")
      for idx in tqdm(range(file_size), desc="PROCESSING", ncols=100, unit=" signal"):
        for sample_idx in range(num_bit_data):
          file.write(str(signal[idx][start[idx] + sample_idx]) + " ")
        file.write("\n")

      file.close()

    except Exception as ex:
      print(ex)

  fileP.close()
