from tqdm import tqdm

from global_vars import *


def read_data_file(file_name, rn_idx):
    try:
        file_signal = open(signal_path + file_name + "_signal", "r")
        file_databit = open(databit_path + file_name + "_databit", "r")
        signal = []
        databit = []

        count = 0
        for idx in tqdm(range(file_num), desc="READING", ncols=100, unit=" signal"):
            sline = file_signal.readline().rstrip()
            dline = file_databit.readline().rstrip()
            if count < len(rn_idx) and idx == rn_idx[count]:
                sdata = sline.split(" ")
                sample = [float(i) for i in sdata]
                signal.append(sample)
                ddata = [int(i) for i in dline]
                databit.append(ddata)
                count += 1

        file_signal.close()
        file_databit.close()

        return signal, databit

    except Exception as ex:
        print("[read_file.py]", end=" ")
        print(ex)

def read_rn_file(file_name):
    try:
        file = open(RN_file_path, "r")

        line = file.readline().rstrip(" \n").split(" ")
        train_idx = [int(i) for i in line]
        line = file.readline().rstrip(" \n").split(" ")
        valid_idx = [int(i) for i in line]
        line = file.readline().rstrip(" \n").split(" ")
        test_idx = [int(i) for i in line]

        train_idx.sort()
        valid_idx.sort()
        test_idx.sort()

        file.close()

        return train_idx, valid_idx, test_idx

    except Exception as ex:
        print("[read_random_index.py]", end=" ")
        print(ex)
