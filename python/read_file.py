from global_vars import *
from tqdm import tqdm


def read_file(file_name):
    try:
        ifile = open(data_path + file_name + "_Isignal", "r")
        qfile = open(data_path + file_name + "_Qsignal", "r")
        input_signals = []

        for n in tqdm(range(file_size), desc=file_name, ncols=80, unit="signal"):
            idata = ifile.readline().rstrip().split(" ")
            qdata = qfile.readline().rstrip().split(" ")
            data = [[float(idata[i]), float(qdata[i])] for i in range(len(idata))]
            input_signals.append([data])

        ifile.close()
        qfile.close()

        file = open(answer_path + file_name + "_databit" + tail, "r")
        answer_signals = []

        for n in range(file_size):
            answer = file.readline().rstrip().split(" ")
            answer = [[float(i), float(i)] for i in answer]
            answer_signals.append([answer])

        file.close()

        file = open(bit_path + file_name + "_databit", "r")
        answer_bits = []

        for n in range(file_size):
            line = file.readline().rstrip()
            if not line:
                break
            bits = [int(i) for i in line]
            answer_bits.append(bits)

        file.close()

        return input_signals, answer_signals, answer_bits

    except Exception as ex:
        print("[read_file.py: " + file_name + "]", end=" ")
        print(ex)
