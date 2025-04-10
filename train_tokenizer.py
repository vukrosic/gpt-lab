"""
Built off of Tiktoken educational implementation 
https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
"""
from __future__ import annotations
import collections
import regex
import tiktoken
import os
import shutil
import argparse
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import pickle
from itertools import chain

import torch
import torch.distributed as dist
from datasets.distributed import split_dataset_by_node
#device = "cuda" if torch.cuda.is_available() else "cpu"
# Check if environment variables are set by torchrun, otherwise default to single GPU
if "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ:
    # Multi-GPU setup with torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    # Single GPU setup
    rank = 0
    world_size = 1
    local_rank = 0
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
print(f"Running with {world_size} GPU(s)")
if world_size == 1:
    print("To run with multiple GPUs where number of GPUs is G, use `torchrun --nproc_per_node=G train_tokenizer.py`")
assert torch.cuda.is_available()
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)

# Initialize distributed process group if using multiple GPUs
if world_size > 1:
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
master_process = (rank == 0)  # this process will do logging, checkpointing etc.


class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, demo: bool = False) -> list[int]:
        """Encodes a string into tokens.

        >>> enc.encode("hello world")
        [388, 372]
        """
        # Use the regex to split the text into (approximately) words
        words = self._pat.findall(text)
        tokens = []
        for word in words:
            # Turn each word into tokens, using the byte pair encoding algorithm
            word_bytes = word.encode("utf-8")
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, demo=demo)
            tokens.extend(word_tokens)
        return tokens

    def decode_bytes(self, tokens: list[int]) -> bytes:
        """Decodes a list of tokens into bytes.

        >>> enc.decode_bytes([388, 372])
        b'hello world'
        """
        return b"".join(self._decoder[token] for token in tokens)

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string.

        Decoded bytes are not guaranteed to be valid UTF-8. In that case, we replace
        the invalid bytes with the replacement character "�".

        >>> enc.decode([388, 372])
        'hello world'
        """
        return self.decode_bytes(tokens).decode("utf-8", errors="replace")

    def decode_tokens_bytes(self, tokens: list[int]) -> list[bytes]:
        """Decodes a list of tokens into a list of bytes.

        Useful for visualising how a string is tokenised.

        >>> enc.decode_tokens_bytes([388, 372])
        [b'hello', b' world']
        """
        return [self._decoder[token] for token in tokens]

    @staticmethod
    def train(training_data: str, vocab_size: int, pat_str: str, demo: bool = False):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size, pat_str=pat_str, demo=demo)
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, demo: bool = False
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if demo: demo_tokens(parts)

        # Iterate over all pairs and find the pair we want to merge the most
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank

        # If there were no pairs we could merge, we're done!
        if min_rank is None:
            break
        assert min_idx is not None

        # Otherwise, merge that pair and leave the rest unchanged. Then repeat.
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2 :]

    tokens = [mergeable_ranks[part] for part in parts]
    return tokens


def slow_merge(words, most_common_pair, token_bytes):
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word) - 1:
            if (word[i], word[i + 1]) == most_common_pair:
                # We found our pair! Merge it
                new_word.append(token_bytes)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        if i == len(word) - 1:
            new_word.append(word[i])
        new_words.append(new_word)
    return new_words


def bpe_train(
    data: str, vocab_size: int, pat_str: str, demo: bool = False
) -> dict[bytes, int]:
    # First, add tokens for each individual byte value
    if vocab_size < 2**8:
        raise ValueError("vocab_size must be at least 256, so we can encode all bytes")
    ranks = {}
    for i in range(2**8):
        ranks[bytes([i])] = i
    int_type = torch.int16 if vocab_size <= 2**15 else torch.int32

    # Splinter up our data into lists of bytes
    words: list[list[bytes]] = [
        [bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, data)
    ]
    # Create a list to store numeric token IDs for tensor operations
    # Initially, these are just byte values (0-255)
    ids_lofl = [[ranks[b] for b in word] for word in words]
    # turn data into parseable tensor - using the token IDs instead of raw bytes
    ids = torch.tensor(
        list(chain.from_iterable(word + [-1] for word in ids_lofl))[:-1], 
        dtype=int_type, device=device)
        # shape (words_in_data * (avg_word_len + 1))\
    
    if master_process and demo:
        # Initialize demo text tokens outside the loop to track changes across iterations
        demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                    f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                    f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
        demo_words = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    progress_bar = tqdm(total=vocab_size - 256, unit="merges")
    # Now, use our data to figure out which merges we should make
    for j in range(256, vocab_size):
        # find most common pair
        pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
        unique, counts = torch.unique(pairs, return_counts=True, dim=1)
            # shapes (2, words_in_data * (avg_word_len + 1)) and (words_in_data * (avg_word_len + 1))
        valid_mask = torch.all(unique != -1, dim=0)
        unique = unique[:, valid_mask]
        counts = counts[valid_mask]
        counts, sort_idx = torch.sort(counts, descending=True)
        pairs_idx = sort_idx[:world_size]#[rank * world_size:(rank + 1) * world_size] # shape (world_size)
        most_common_pairs_local = unique[:, pairs_idx] # (2, world_size)
        counts_local = counts[:world_size]#[:pairs_idx] # (world_size)
        most_common_pairs_global = torch.zeros((2,world_size ** 2), dtype=torch.float32, device=device)
        counts_global = torch.zeros(world_size ** 2, dtype=torch.float32, device=device)
        most_common_pairs_global[:, rank * world_size:(rank + 1) * world_size] = most_common_pairs_local.to(torch.float32)
        counts_global[rank * world_size:(rank + 1) * world_size] = counts_local.to(torch.float32)
        
        if world_size > 1:
            dist.all_reduce(most_common_pairs_global, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts_global, op=dist.ReduceOp.SUM)

        # First, get unique pairs and their counts from the combined data
        pairs_global = most_common_pairs_global.t()  # Shape: [world_size^2, 2]
        unique_pairs, inverse_indices = torch.unique(pairs_global, dim=0, return_inverse=True)

        # Sum the counts for each unique pair
        sum_counts = torch.zeros(unique_pairs.size(0), dtype=torch.float, device=device)
        sum_counts.scatter_add_(0, inverse_indices, counts_global.float())

        # Count occurrences of each unique pair
        pair_occurrences = torch.bincount(inverse_indices)

        # Find the maximum occurrence count
        max_occurrence = torch.max(pair_occurrences)

        # Create a mask for pairs with the maximum occurrence count
        max_occurrence_mask = (pair_occurrences == max_occurrence)

        # Filter to only consider pairs with the maximum occurrence count
        filtered_sum_counts = sum_counts[max_occurrence_mask]
        filtered_unique_pairs = unique_pairs[max_occurrence_mask]

        # Find the pair with the largest count among the filtered pairs
        max_index = torch.argmax(filtered_sum_counts)
        best_pair = filtered_unique_pairs[max_index].cpu().numpy() # Shape: (2)

        # Map token IDs back to the corresponding byte sequences
        # Using the dictionary in reverse to get the bytes corresponding to these IDs
        best_bytes = [None, None]
        for bytes_token, id_token in ranks.items():
            if id_token == best_pair[0]:
                best_bytes[0] = bytes_token
            if id_token == best_pair[1]:
                best_bytes[1] = bytes_token
        token_bytes = best_bytes[0] + best_bytes[1]
        new_token_id = len(ranks)
        # Add the new token!
        ranks[token_bytes] = new_token_id

        # Now merge that most common pair in all the words
        pair_mask = (pairs[0] == best_pair[0]) & (pairs[1] == best_pair[1]) 
        ids[:-1][pair_mask] = new_token_id
        ids[1:][pair_mask] = -2
        keep_mask = (ids != -2)
        ids = ids[keep_mask]

        if demo and master_process:
            # Also apply the same merge to our demo text
            demo_words = slow_merge(demo_words, tuple(best_bytes), token_bytes)

            # See the intermediate merges play out!
            if j % 500 == 0 or j in [256, vocab_size - 1]:
                print(f"The most common pair {best_pair[0]} + {best_pair[1]} "
                        f"which makes {token_bytes} our {len(ranks)}th token")
                # Flatten the demo words into a single list of tokens for visualization
                demo_tokens = [token for word in demo_words for token in word]
                visualise_tokens(demo_tokens)
                print("\n")

        if master_process:
            progress_bar.update(1)

    return ranks


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # demo the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")


def fetch_fineweb_data(max_chars: int = 2**22):
    # Use the 10B version of FineWeb; not edu since the former should have more diverse tokens
    dataset = load_dataset("HuggingFaceFW/fineweb", 
                           name="sample-10BT", 
                           split="train", 
                           streaming=True)
    if world_size > 1:
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    
    text_data = []
    doc_lengths = []
    tot_len = 0
    for i, item in enumerate(dataset):
        tot_len += len(item["text"])
        if tot_len >= max_chars:
            tot_len -= len(item["text"])
            break
        text_data.append(item["text"])
        doc_lengths.append(len(item["text"]))
    
    # Show statistics if requested
    if rank == 0:
        print(f"\nDataset Statistics for GPU {rank}:"
            f"\nTotal documents: {len(text_data)}"
            f"\nTotal characters: {sum(doc_lengths):,}"
            f"\nAverage document length: {np.mean(doc_lengths):.1f} characters"
            f"\nMedian document length: {np.median(doc_lengths):.1f} characters"
            f"\nShortest document: {min(doc_lengths)} characters"
            f"\nLongest document: {max(doc_lengths):,} characters"
            f"\nStandard deviation: {np.std(doc_lengths):.1f} characters")
    
    # Join all text data into a single string
    return "\n".join(text_data)


def train_simple_encoding(sample_size: int, vocab_size: int, demo: bool = False):
    """
    Train a custom BPE tokenizer using FineWeb data.
    
    Args:
        sample_size: maximum number of characters to include in data
        vocab_size: Size of the vocabulary to train
        demo: Visualization mode for BPE training process
    
    Returns:
        The trained tokenizer
    """
    data = fetch_fineweb_data(max_chars=sample_size)

    #gpt2_pattern = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    gpt4_pattern = (
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    )
    enc = SimpleBytePairEncoding.train(data, vocab_size=vocab_size, pat_str=gpt4_pattern, demo=demo)
    
    # Test the tokenizer with a simple example
    test_str = f"hello world rank={rank}"
    tokens = enc.encode(test_str)
    # Verify encoding-decoding roundtrips correctly
    decoded = enc.decode(tokens)
    assert decoded == test_str, f"Decoding failed: expected '{test_str}' but got '{decoded}'"
    decoded_bytes = enc.decode_bytes(tokens)
    assert decoded_bytes == test_str.encode('utf-8'), \
        f"Bytes decoding failed: expected {test_str.encode('utf-8')} but got {decoded_bytes}"
    
    return enc


def save_tokenizer(enc, name, vocab_size, sample_size):
    """Save the tokenizer for later use"""
    # Ensure the directory exists
    os.makedirs('tokenizers', exist_ok=True)
    
    # Construct the filename
    full_filename = f"tokenizers/{name}_v{vocab_size}_n{sample_size}.pkl"
    
    # Read the content of train_tokenizer.py
    with open(__file__, 'r') as f:
        script_content = f.read()
    
    # Prepare the tokenizer data
    tokenizer_data = {
        "pat_str": enc.pat_str,
        "mergeable_ranks": enc.mergeable_ranks,
        "script_content": script_content  # Add the script content for backup
    }
    
    # Save the tokenizer data
    with open(full_filename, 'wb') as f:
        pickle.dump(tokenizer_data, f)
    
    print(f"Tokenizer saved to {full_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("-n", "--samples", type=int, default=2**27, 
        help="Maximum number of text characters to use PER GPU for training (default 2^27 should fit on 8gb of VRAM)")
    parser.add_argument("-v", "--vocabsize", type=int, default=50256, 
        help="Size of the vocabulary to train (default 50256; same as GPT2 minus <|endoftext|>)")
    parser.add_argument("-f", "--name", type=str, default="gpt4regex", 
        help="Filename prefix to save the tokenizer (default 'gpt4regex')")
    parser.add_argument("--demo", action="store_true", default=False, 
        help="Visualize tokenization during training")
    args = parser.parse_args()

    # Train the tokenizer
    enc = train_simple_encoding(
        sample_size=args.samples,
        vocab_size=args.vocabsize,
        demo=args.demo
    )
    
    # Save the tokenizer
    if master_process:
        save_tokenizer(enc, args.name, args.vocabsize, args.samples * world_size)
    