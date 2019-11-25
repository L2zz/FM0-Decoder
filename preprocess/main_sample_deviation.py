from tqdm import tqdm

from read_file import read_file
from standardize_sample import standardize_sample

import global_vars
target_path = global_vars.target_path
file_size = global_vars.file_size
file_name_list = global_vars.file_name_list



if __name__ == "__main__":
  fileA = open(target_path + "average_std", "w")
  fileV = open(target_path + "variation_std", "w")

  for file_name in file_name_list:
    try:
      print("\n\n\t*** " + file_name + " ***")
      fileA.write("\n\n\t*** " + file_name + " ***\n")
      fileV.write("\n\n\t*** " + file_name + " ***\n")
      signal = read_file(file_name)
      signal = standardize_sample(signal)
      sample_size = len(signal[0])

      for idx in tqdm(range(file_size), desc="PROCESSING", ncols=100, unit=" signal"):
        avg = 0
        for sample in signal[idx]:
          avg += sample
        avg /= sample_size
        fileA.write(str(avg) + " ")

        var = 0
        for sample in signal[idx]:
          var += pow(sample - avg, 2)
        var /= sample_size
        fileV.write(str(var) + " ")

    except Exception as ex:
      print(ex)

  fileA.close()
  fileV.close()
