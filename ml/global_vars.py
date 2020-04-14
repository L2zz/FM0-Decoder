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
enc256_path = data_dir + "databit_enc256/"


execute_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# file_name = execute_time
# file_name = "student3000"
# file_name = "teacher3000"
file_name = "kd"
log_path = "../log/"
log_file_path = log_path + file_name
model_path = "../model/"
teacher_file_path = model_path + "teacher3000"
model_file_path = model_path + file_name

RN_tail = 1
RN_file_name = "RN_" + str(file_num) + "_" + str(RN_tail)
RN_file_path = data_dir + "random_index/" + RN_file_name

"""
For model
"""
epochs = 200
lr = 0.001
batch_size = 64
isEarlyStop = True
isBestSave = False

alpha = 0.0
