import torch
a = torch.tensor([0,1,2,3,4,5,6,6,6])
b = torch.unique(a)
print(b)