import torch

# Define the tensors
most_common_pairs = torch.tensor(
    [
        [4, 8, 2, 3, 4, 3, 4, 4, 2, 2],
        [1, 3, 9, 5, 1, 5, 1, 2, 9, 9]
    ]
)
counts = torch.tensor([8, 3, 6, 12, 2, 15, 9, 6, 10, 9])

# Transpose the pairs to make each column a pair
pairs = most_common_pairs.t()

# Find unique pairs and their inverse indices
unique_pairs, inverse_indices = torch.unique(pairs, dim=0, return_inverse=True)

# Sum the counts for each unique pair
sum_counts = torch.zeros(unique_pairs.size(0), dtype=torch.float)
sum_counts.scatter_add_(0, inverse_indices, counts.float())

# Count occurrences of each unique pair
pair_occurrences = torch.bincount(inverse_indices)

# Find the maximum occurrence count
max_occurrence = torch.max(pair_occurrences)

# Create a mask for pairs with the maximum occurrence count
max_occurrence_mask = (pair_occurrences == max_occurrence)

# Calculate the average count for each unique pair
average_counts = sum_counts / pair_occurrences

# Filter to only consider pairs with the maximum occurrence count
filtered_average_counts = average_counts[max_occurrence_mask]
filtered_unique_pairs = unique_pairs[max_occurrence_mask]

# Find the index of the pair with the largest average count among the filtered pairs
max_index = torch.argmax(filtered_average_counts)
max_pair = filtered_unique_pairs[max_index]
max_average_count = filtered_average_counts[max_index]

print(f"The pair with the largest average count among those with the most duplicates ({max_occurrence.item()} occurrences) is {max_pair.tolist()} with an average count of {max_average_count.item()}.")
