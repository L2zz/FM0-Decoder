from global_vars import *
from tqdm import tqdm


def read_test_file(file_name):
    try:
        rnidx_file = open(RN_path, "r")
        rnidx_file.readline()
        rnidx_file.readline()

        test_rn_idx = rnidx_file.readline().rstrip().split(" ")
        test_rn_idx = [int(i) for i in test_rn_idx]
        test_rn_idx.sort()

        ifile = open(data_path + file_name + "_Isignal", "r")
        qfile = open(data_path + file_name + "_Qsignal", "r")
        ansfile = open(bit_path + file_name + "_databit", "r")

        input_signals = []
        answer_bits = []

        count = 0
        for n in tqdm(range(len(test_rn_idx)), desc=file_name, ncols=80, unit="signal"):
            target_idx = test_rn_idx[n]
            while target_idx != count:
                count += 1
                ifile.readline()
                qfile.readline()
                ansfile.readline()
            count += 1
            iline = ifile.readline().rstrip()
            qline = qfile.readline().rstrip()
            aline = ansfile.readline().rstrip()

            idata = iline.split(" ")
            qdata = qline.split(" ")
            data = [[float(idata[i]), float(qdata[i])] for i in range(len(idata))]
            input_signals.append([data])

            answer = [int(i) for i in aline]
            answer_bits.append(answer)

        ifile.close()
        qfile.close()
        ansfile.close()

        return input_signals, answer_bits

    except Exception as ex:
        print("[read_file.py: " + file_name + "]", end=" ")
        print(ex)
