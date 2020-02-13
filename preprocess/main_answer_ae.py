from tqdm import tqdm
from global_vars import *

if __name__ == "__main__":
    for idx in tqdm(range(len(file_name_list)), desc="PROCESSING", ncols=80, unit=" file"):
        file_name = file_name_list[idx]

        try:
            fileR = open(target_path + file_name + "_answer_enc256", "r")
            fileW = open(target_path + file_name + "_answer_ae", "w")

            half_extra_length = int(extra_length / 2)

            for n in range(file_size):
                line = fileR.readline().rstrip("\n")
                data = [int(i) for i in line]

                for _ in range(half_extra_length):
                    fileW.write("-1.0 ")

                for bit in data:
                    if bit == 0:
                        for _ in range(num_half_bit):
                            fileW.write("-1.0 ")
                    else:
                        for _ in range(num_half_bit):
                            fileW.write("1.0 ")

                for _ in range(half_extra_length):
                    fileW.write("-1.0 ")
                fileW.write("\n")

            fileR.close()
            fileW.close()

        except Exception as ex:
            continue
