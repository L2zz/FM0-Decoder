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
            idata.insert(0, idata[0])
            idata.insert(-1, idata[-1])
            qdata.insert(0, qdata[0])
            qdata.insert(-1, qdata[-1])
            data = [[float(idata[i]), float(qdata[i])] for i in range(len(idata))]
            input_signals.append([data])

        ifile.close()
        qfile.close()

        return input_signals

    except Exception as ex:
        print("[read_file.py: " + file_name + "]", end=" ")
        print(ex)
