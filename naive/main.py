import time

import numpy as np
from tqdm import tqdm

from global_vars import *
from read_file import read_data_file, read_rn_file
from decode_data import decode_data
from ExecutionTime import ExecutionTime
from standardize_sample import standardize_sample


tot_test_file_num = len(file_name_list) * test_file_num

if __name__ == "__main__":
    tot_time = ExecutionTime("TOTAL")
    log_file = open(log_file_path, "w")

    rn_file = open(RN_file_path, "r")
    _, _, rn_idx = read_rn_file(rn_file)

    success = 0
    success_bit = np.zeros(bit_data + 1)
    tot_decode_time = 0
    for file_name in file_name_list:
        try:
            file_time = ExecutionTime(file_name)
            print("\n\n\t*** " + file_name + " ***")
            log_file.write("\t*** " + file_name + " ***\n")
            signal, databit = read_data_file(file_name, rn_idx)
            signal = standardize_sample(signal)

            cur_success_bit = np.zeros(bit_data + 1)
            cur_success = 0
            num_signal = len(signal)
            decode_time_per_file = 0
            for idx in tqdm(range(num_signal), desc="DECODING", ncols=100, unit=" signal"):
                begin = time.time()
                decoded_bit = decode_data(signal[idx])
                end = time.time()

                count = 0
                for bit_idx in range(bit_data):
                    if decoded_bit[bit_idx] == databit[idx][bit_idx]:
                        count += 1
                if count == bit_data:
                    cur_success += 1
                    success += 1
                cur_success_bit[count] += 1
                success_bit[count] += 1
                decode_time_per_file += end - begin
            tot_decode_time += decode_time_per_file
            print('{:.6f}'.format(decode_time_per_file/num_signal))

            print("\n\tRESULT: " + str(cur_success) +
                  " / " + str(test_file_num), end=" ")
            print("(" + str(round(100 * (float(cur_success) / test_file_num), 2)) + "%)")
            file_time.stop(False)

            log_file.write("RESULT: " + str(cur_success) + " / " + str(test_file_num))
            log_file.write(
                "(" + str(round(100 * (float(cur_success) / test_file_num), 2)) + "%)\n")
            file_time.print_file(log_file)
            for data in cur_success_bit:
                log_file.write(str(int(data)) + " ")
            log_file.write("\n\n\n")

        except Exception as ex:
            print("[main.py]", end=" ")
            print(ex)
            file_time.stop(False)
    print('{:.6f}'.format(tot_decode_time/(len(file_name_list)*600)))

    print("\n\n\t*** SUMMARY ***")
    print("\tTOTALRESULT: " + str(success) +
          " / " + str(tot_test_file_num), end=" ")
    print("(" + str(round(100 * (float(success) / tot_test_file_num), 2)) + "%)")
    tot_time.stop(True)
    print("\n")

    log_file.write("\t*** SUMMARY ***\n")
    log_file.write("RESULT: " + str(success) + " / " + str(tot_test_file_num))
    log_file.write(
        "(" + str(round(100 * (float(success) / tot_test_file_num), 2)) + "%)\n")
    tot_time.print_file(log_file)
    for data in success_bit:
        log_file.write(str(int(data)) + " ")

    log_file.close()
