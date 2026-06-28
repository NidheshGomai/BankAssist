import torch
print("PyTorch Version:", torch.__version__)
print("Is CUDA active?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Target GPU:", torch.cuda.get_device_name(0))