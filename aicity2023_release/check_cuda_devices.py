import torch

print("cuda available:",torch.cuda.is_available())  # Returns True if CUDA is available and can be used
print("devices count:",torch.cuda.device_count())  # Returns the number of CUDA devices available
print("current device:",torch.cuda.current_device())  # Returns the current device
print("printing devices")
for i in range(torch.cuda.device_count()):
    print(f"{i}: ",torch.cuda.get_device_name(i))  # Returns the name of the first CUDA device
