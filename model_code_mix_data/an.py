import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Is GPU available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")