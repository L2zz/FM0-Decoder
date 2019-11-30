# select model
tail = "_enc256"
#tail = "_enc128"



# data
num_half_bit = 25
bit_preamble = 6
bit_data = 128
bit_extra = 12

#data_path = "data/190917_nopadding/"
data_path = "data/191019_index_std/"
#data_path = "data/191019_padding_std/"

file_name_list = []
'''
for a in ["50", "100", "150", "200", "250", "300", "350", "400"]:
  for b in ["l", "r"]:
    for c in ["20", "60", "100"]:
      file_name_list.append(a + "_" + b + c)

'''

"""
for a in ["100", "200", "300", "400"]:
    for b in ["0", "l100", "r100"]:
        for c in ["0", "45", "90", "135"]:
           file_name_list.append(a + "_" + b + "_" + c)
"""

'''
# good group
file_name_list.append("100_0_0")
file_name_list.append("100_0_45")
file_name_list.append("100_0_90")
file_name_list.append("100_0_135")
file_name_list.append("100_l100_0")
file_name_list.append("100_l100_135")
file_name_list.append("100_r100_0")
file_name_list.append("100_r100_45")
file_name_list.append("200_0_0")
file_name_list.append("200_0_45")
file_name_list.append("200_0_135")
file_name_list.append("200_l100_0")
file_name_list.append("200_l100_90")
file_name_list.append("200_l100_135")
file_name_list.append("200_r100_45")
file_name_list.append("200_r100_135")
file_name_list.append("300_0_0")
file_name_list.append("300_0_135")
file_name_list.append("300_r100_45")
file_name_list.append("300_r100_90")
file_name_list.append("300_r100_135")
file_name_list.append("400_l100_135")
file_name_list.append("400_r100_0")
file_name_list.append("400_r100_45")
'''
'''
# bad group
file_name_list.append("100_l100_45")
file_name_list.append("100_l100_90")
file_name_list.append("100_r100_90")
file_name_list.append("100_r100_135")
file_name_list.append("200_0_90")
file_name_list.append("200_l100_45")
file_name_list.append("200_r100_0")
file_name_list.append("200_r100_90")
file_name_list.append("300_0_45")
file_name_list.append("300_0_90")
file_name_list.append("300_l100_0")
file_name_list.append("300_l100_45")
file_name_list.append("300_l100_90")
file_name_list.append("300_l100_135")
file_name_list.append("300_r100_0")
file_name_list.append("400_0_0")
file_name_list.append("400_0_45")
file_name_list.append("400_0_90")
file_name_list.append("400_0_135")
file_name_list.append("400_l100_0")
file_name_list.append("400_l100_45")
file_name_list.append("400_l100_90")
file_name_list.append("400_r100_90")
file_name_list.append("400_r100_135")
'''

file_name_list.append("200_l100_0")
file_name_list.append("100_0_90")
file_name_list.append("200_0_0")

#file_name_list.append("100_l100_135")
#file_name_list.append("100_0_45")
#file_name_list.append("100_0_135")


# time & folder_path
import datetime
execute_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
model_path = "model/"
model_full_path = model_path + execute_time
log_path = "log/"
log_full_path = log_path + execute_time



# define set
#file_size = 2000
file_size = 3000
ratio_test_per_train = 0.2
ratio_validation_per_train = 0.2

test_size = int(file_size * ratio_test_per_train)
validation_size = int(file_size * (1-ratio_test_per_train) * ratio_validation_per_train)
train_size = int(file_size * (1-ratio_test_per_train) * (1-ratio_validation_per_train))



# Autoencoder
model_tail = ""
