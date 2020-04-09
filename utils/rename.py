import os

model_path = "../model/"
log_path = "../log/"

if __name__ == "__main__":
  try:
    print("\n\n\n" + str(os.listdir(model_path)))
    old_name = input("\n\n\nInput the model name you want to rename: ").rstrip("\n")
    if not os.path.exists(model_path + old_name):
      raise NameError("Model \"" + old_name + "\" does not exist")

    new_name = input("Input the new name: ").rstrip("\n")
    os.rename(model_path + old_name, model_path + new_name)
    os.rename(log_path + old_name, log_path + new_name)

  except Exception as ex:
    print("[rename.py]", end=" ")
    print(ex)
