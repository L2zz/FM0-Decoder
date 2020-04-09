import datetime

"""
General variables
"""
num_half_bit = 25
bit_preamble = 6
bit_data = 128
bit_extra = 12

file_num = 3000
test_file_num = file_num * 0.2
file_name_list = []

for a in ["100", "200", "300", "400"]:
    for b in ["0", "l100", "r100"]:
        for c in ["0", "45", "90", "135"]:
            file_name_list.append(a + "_" + b + "_" + c)

"""
For file IO
"""
data_dir = "../../sin-decode/data/"
signal_path = data_dir + "Amp_zerostd_clip/"
databit_path = data_dir + "databit/"

execute_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_file_path = "log/" + execute_time

RN_tail = 1
RN_file_name = "RN_" + str(file_num) + "_" + str(RN_tail)
RN_file_path = data_dir + "random_index/" + RN_file_name
