import torch

a = torch.ones((2,5,4))
a.shape
print(a.sum(axis=1,keepdims=True).shape)
a.sum(axis=1).shape


# Path: train.py

b = 3


print("hello world")