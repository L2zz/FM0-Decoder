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
            # idata.insert(0, idata[0])
            # idata.insert(-1, idata[-1])
            # qdata.insert(0, qdata[0])
            # qdata.insert(-1, qdata[-1])
            data = [[float(data[i])] for i in range(len(data))]
            input_signals.append([data])

        file.close()

        file = open(answer_path + file_name + "_databit" + tail, "r")
        answer_signals = []

        for n in range(file_size):
            answer = file.readline().rstrip().split(" ")
            del answer[0]
            del answer[-1]
            # answer.insert(0, answer[0])
            # answer.insert(-1, answer[-1])
            answer = [[float(i)] for i in answer]
            answer_signals.append([answer])

        file.close()

        file = open(bit_path + file_name + "_databit", "r")
        answer_bits = []

        for n in range(file_size):
            line = file.readline().rstrip()
            if not line:
                break
            data = [int(i) for i in line]
            answer_bits.append(data)

        file.close()

        return input_signals, answer_signals, answer_bits

    except Exception as ex:
        print("[read_file.py: " + file_name + "]", end=" ")
        print(ex)
