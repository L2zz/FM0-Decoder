from tqdm import tqdm

import global_vars
source_path = global_vars.source_path
target_path = global_vars.target_path
file_size = global_vars.file_size
file_name_list = global_vars.file_name_list



if __name__ == "__main__":
  for idx in tqdm(range(len(file_name_list)), desc="PROCESSING", ncols=80, unit=" file"):
    file_name = file_name_list[idx]

    try:
      fileR = open(source_path + file_name + "_databit", "r")
      fileW = open(target_path + file_name + "_answer_enc256", "w")

      for n in range(file_size):
        fileW.write("110100100011") # preamble

        line = fileR.readline().rstrip("\n")
        data = [int(i) for i in line]

        conv = []
        level = -1
        for sample in data:
          if sample == 1:
            conv.append(level)
            conv.append(level)
            level *= -1
          else:
            conv.append(level)
            conv.append(level * -1)

        for sample in conv:
          if sample == -1:
            fileW.write("0")
          else:
            fileW.write("1")
        fileW.write("\n")

      fileR.close()
      fileW.close()

    except Exception as ex:
      continue
