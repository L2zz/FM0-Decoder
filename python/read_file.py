from global_vars import *
from tqdm import tqdm


def read_file(file_name):
    try:
        file = open(data_path + file_name + "_signal", "r")
        input_signals = []

        for n in tqdm(range(file_size), desc=file_name, ncols=80, unit="signal"):
            data = file.readline().rstrip().split(" ")
            del data[0]
            del data[-1]
            data = [[float(data[i])] for i in range(len(data))]
            input_signals.append([data])

        file.close()

        return input_signals

    except Exception as ex:
        print("[read_file.py: " + file_name + "]", end=" ")
        print(ex)
