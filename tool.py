import torch
import torch.nn as nn
from thop import profile

from hyperparameter import hyperparameter
from model import AttentionDTI

hp = hyperparameter()
model = AttentionDTI(hp)
input_tensor1 = torch.randint(0, 65, (1, 100))
input_tensor2 = torch.randint(0, 26, (1, 1000))

flops, params = profile(model, inputs=(input_tensor1.to(torch.int64), input_tensor2.to(torch.int64)))
print(f"Model has {params} trainable parameters.")
print(f"FLOPs: {flops}")