# select model
tail = "_enc256"
sample_type = "org"
#sample_type = "128bit"
repetition = 1



# Autoencoder
isEarlyStop = True
dropout_rate = 0

if sample_type == "org":
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



# data
num_half_bit = 25
bit_preamble = 6
bit_data = 128
bit_extra = 12

if sample_type == "org":
  data_path = "data/D_org_std/"
elif sample_type == "128bit":
  data_path = "data/D_128bit_std_fitIndex/"

RN_tail = "_RN1"
RN_set = "RN_3000_1"
RN_path = "data/random_index/" + RN_set



# time & folder_path
import datetime
execute_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
model_path = "model/"
model_full_path = model_path + execute_time
log_path = "log/"
log_full_path = log_path + execute_time



# define set
file_size = 3000
ratio_test_per_train = 0.2
ratio_validation_per_train = 0.2

test_size = int(file_size * ratio_test_per_train)
validation_size = int((file_size - test_size) * ratio_validation_per_train)
train_size = file_size - test_size - validation_size



# select file
file_name_list = []

for a in ["100", "200", "300", "400"]:
    for b in ["0", "l100", "r100"]:
        for c in ["0", "45", "90", "135"]:
           file_name_list.append(a + "_" + b + "_" + c)


# good group
#file_name_list = ["100_0_0", "100_0_45", "100_0_90", "100_0_135", "100_l100_0", "100_l100_135", "100_r100_0", "100_r100_45",
#                  "200_0_0", "200_0_45", "200_0_135", "200_l100_0", "200_l100_90", "200_l100_135", "200_r100_45", "200_r100_135",
#                  "300_0_0", "300_0_135", "300_r100_45", "300_r100_90", "300_r100_135", "400_l100_135", "400_r100_0", "400_r100_45"]

# bad group
#file_name_list = ["100_l100_45", "100_l100_90", "100_r100_90", "100_r100_135", "200_0_90", "200_l100_45", "200_r100_0", "200_r100_90",
#                  "300_0_45", "300_0_90", "300_l100_0", "300_l100_45", "300_l100_90", "300_l100_135", "300_r100_0", "400_0_0",
#                  "400_0_45", "400_0_90", "400_0_135", "400_l100_0", "400_l100_45", "400_l100_90", "400_r100_90", "400_r100_135"]

# rank 1 ~ 3
#file_name_list.append("200_l100_0")
#file_name_list.append("100_0_90")
#file_name_list.append("200_0_0")

# rank 4 ~ 6
#file_name_list.append("100_l100_135")
#file_name_list.append("100_0_45")
#file_name_list.append("100_0_135")
