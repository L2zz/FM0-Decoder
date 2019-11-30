from tqdm import tqdm

from read_file import read_file
from standardize_sample import standardize_sample
from detect_preamble import detect_preamble

import global_vars
target_path = global_vars.target_path
num_bit_data = global_vars.bit_data * 2 * global_vars.num_half_bit
file_size = global_vars.file_size
file_name_list = global_vars.file_name_list



if __name__ == "__main__":
  for file_name in file_name_list:
    try:
      print("\n\n\t*** " + file_name + " ***")
      signal = read_file(file_name)
      signal = standardize_sample(signal)

      file = open(target_path + file_name + "_sample", "w")
      for idx in tqdm(range(file_size), desc="PROCESSING", ncols=100, unit=" signal"):
        for sample_idx in range(len(signal[idx])):
          file.write(str(signal[idx][sample_idx]) + " ")
        file.write("\n")

      file.close()

    except Exception as ex:
      print(ex)
