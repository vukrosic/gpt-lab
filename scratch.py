from itertools import chain

a = [[1,2,3], [4,5,6], [7,8,9]]

# Flatten the list of lists with -1 in between each sub-list
result = list(chain.from_iterable(sublist + [-1] for sublist in a))[:-1]

print(result)