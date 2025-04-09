from itertools import chain
import torch

a = [[1,2,3], [4,5,6], [7,8,9]]

# Flatten the list of lists with -1 in between each sub-list
result = list(chain.from_iterable(sublist + [-1] for sublist in a))[:-1]

print(result)

b = torch.tensor([1,4,7,4,3,5,8,7,5,3,2,6])
print(torch.sort(b, descending=True))