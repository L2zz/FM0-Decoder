"""
Define network structure
"""
# select model
import datetime
tail = "_enc256"
type = "_vgg"
sample_type = "org"
repetition = 1

# Network
isCNN = True
isEarlyStop = True
dropout_rate = 0.

if isCNN:
    learning_rate = 0.001
    learning_epoch = 100
    batch_size = 200

    size_kernel = 5
    size_pool = 2
    padding = "same"

    size_input_layer = 7300

    size_conv_layer1 = 32
    size_conv_layer2 = 32

    size_conv_layer3 = 64
    size_conv_layer4 = 64

    size_conv_layer5 = 128
    size_conv_layer6 = 128

    size_conv_layer7 = 256
    size_conv_layer8 = 256

    size_conv_layer9 = 256
    size_conv_layer10 = 256

    size_fc_layer1 = 1024
    size_fc_layer2 = 1024

    size_output_layer = 268

elif sample_type == "org":
    learning_rate = 0.00005
    learning_epoch = 100

    size_input_layer = 7300
    size_hidden_layer = 3650
    size_hidden_layer2 = 3650
    size_hidden_layer3 = 6700
    size_hidden_layer4 = 6700
    size_hidden_layer5 = 1340
    size_hidden_layer6 = 1340
    size_output_layer = 268 * repetition

elif sample_type == "128bit":
    learning_rate = 0.001
    learning_epoch = 50

    size_input_layer = 6400
    size_hidden_layer = 3200
    size_hidden_layer2 = 256
    size_hidden_layer3 = 3200
    size_hidden_layer4 = 256
    size_hidden_layer5 = 3200
    size_hidden_layer6 = 0  # not used
    size_output_layer = 256 * repetition


"""
Define general data info
"""
# data
num_half_bit = 25
bit_preamble = 6
bit_data = 128
bit_extra = 12

"""
Set data path
"""
if sample_type == "org":
    data_path = "data/D_org_std/"
elif sample_type == "128bit":
    data_path = "data/D_128bit_std_fitIndex/"

"""
For random indexing
"""
RN_tail = "_RN2"
RN_set = "RN_1000_1"
RN_path = "data/random_index/" + RN_set

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
file_size = 1000
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
