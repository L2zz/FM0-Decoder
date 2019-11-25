from tqdm import tqdm

from read_file import read_file
from standardize_sample import standardize_sample
from detect_preamble import detect_preamble

import global_vars
target_path = global_vars.target_path
file_size = global_vars.file_size
file_name_list = global_vars.file_name_list



if __name__ == "__main__":
  file = open(target_path + "preamble_index", "w")

  for file_name in file_name_list:
    try:
      print("\n\n\t*** " + file_name + " ***")
      file.write("\n\n\t*** " + file_name + " ***\n")
      signal = read_file(file_name)
      signal = standardize_sample(signal)

      for idx in tqdm(range(file_size), desc="PROCESSING", ncols=100, unit=" signal"):
        start, state = detect_preamble(signal[idx])
        file.write(str(start) + " ")

    except Exception as ex:
      print(ex)

  file.close()
