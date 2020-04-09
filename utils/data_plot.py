"""
module for reading data in target dir
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


def read_file(file, max_num_sig=1000, start_idx=0):
    """
    Read file and return written signals
    @param
        file: target file name
        max_num_sig: maixmum number of signals to get
        start_idx: start index of signal to get
    @return
        Signals: array to save signals
    """
    with open(file, "r") as file_reader:
        ln = 0
        try:
            for _ in range(start_idx):
                file_reader.readline()
                ln += 1
            for _ in tqdm(range(max_num_sig), desc=file, ncols=80):
                line = file_reader.readline()
                ln += 1
                str_values = line.split(' ')
                str_values.remove('\n')  # Remove newline chararcter at last
                values = []
                for str_value in str_values:
                  values.append(float(str_value))
                print(len(values))
                plt.plot(values)
                plt.show()
        except Exception as ex:
            pass


if __name__ == "__main__":

    TARGET = sys.argv[1]
    MAX_NUM_SIG = int(sys.argv[2])

    read_file(TARGET, MAX_NUM_SIG)
