import os

import torch

print(torch.__file__)
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("cudnn version", torch.backends.cudnn.version())
