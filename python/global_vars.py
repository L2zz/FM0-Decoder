"""
Define network structure
"""
# select model
import datetime
tail = "_6850"

# Network
isEarlyStop = True
isBestSave = False
dropout_rate = 0.

learning_rate = 0.001
learning_epoch = 50
batch_size = 100

size_kernel = 5
size_pool = (1, 2)
padding = "same"

size_input_layer = 6850

size_conv_layer1 = 32
size_conv_layer2 = 32

size_conv_layer3 = 64
size_conv_layer4 = 64

size_conv_layer5 = 128
size_conv_layer6 = 128

size_deconv_layer1 = size_conv_layer6
size_deconv_layer2 = size_conv_layer5
size_deconv_layer3 = size_conv_layer4
size_deconv_layer4 = size_conv_layer3
size_deconv_layer5 = size_conv_layer2
size_deconv_layer6 = size_conv_layer1

size_deconv_layer7 = 1

size_output_layer = size_input_layer


"""
Define general data info
"""
# data
num_half_bit = 25
bit_preamble = 6
bit_data = 128
bit_extra = 12
extra_length = bit_extra * num_half_bit * 2

"""
Set data path
"""
data_path = "../sin-decode/data/Amp_zerostd_clip/"
answer_path = "../sin-decode/data/databit_6850/"
bit_path = "../sin-decode/data/databit/"

"""
For random indexing
"""
RN_tail = "_RN3"
RN_set = "RN_200_1"
RN_path = "../sin-decode/data/random_index/" + RN_set

"""
For logging
"""
# time & folder_path
execute_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
model_path = "model/"
model_full_path = model_path + execute_time
log_path = "log/"
log_full_path = log_path + execute_time


"""
Define train/test set
"""
# define set
file_size = 200
ratio_test_per_train = 0.2
ratio_validation_per_train = 0.2

test_size = int(file_size * ratio_test_per_train)
validation_size = int((file_size - test_size) * ratio_validation_per_train)
train_size = file_size - test_size - validation_size


"""
List of file set to read
"""
# select file
file_name_list = []

for a in ["100", "200", "300", "400"]:
    for b in ["0", "l100", "r100"]:
        for c in ["0", "45", "90", "135"]:
            file_name_list.append(a + "_" + b + "_" + c)
