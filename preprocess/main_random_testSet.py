from tqdm import tqdm
from global_vars import *



if __name__ == "__main__":
  try:
    file = open(RN_path, "r")
    line = file.readline()
    line = file.readline()
    line = file.readline().rstrip(" \n").split(" ")
    test_idx = [int(i) for i in line]
    test_idx.sort()
    file.close()

  except Exception as ex:
    print("[main_random_testSet: read_index]", end=" ")
    print(ex)



  try:
    # _target_name = "_sample"
    _target_name = "_answer_ae"

    for file_name in file_name_list:
      try:
        fileR = open(target_path + file_name + _target_name, "r")
        fileW = open(target_path + file_name + _target_name + RN_tail, "w")
        idx_index = 0

        for idx in tqdm(range(file_size), desc=file_name, ncols=100, unit=" signal"):
          if idx == test_idx[idx_index]:
            fileW.write(fileR.readline())
            idx_index += 1
            if idx_index == test_set_size:
              break
          else:
            fileR.readline()

        fileR.close()
        fileW.close()

      except Exception as ex:
        print("[main_random_testSet: " + file_name + "]", end=" ")
        print(ex)

  except Exception as ex:
    print("[main_random_testSet: processing]", end=" ")
    print(ex)
