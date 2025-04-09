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
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    m = max([len(word) for word in words])
    # Create a list to store numeric token IDs for tensor operations
    # Initially, these are just byte values (0-255)
    ids_lofl = [[ranks[b] for b in word] for word in words]
    # Flatten the list of lists with -1 in between each sub-list
    ids_list = list(chain.from_iterable(word + [-1] for word in ids_lofl))[:-1]
    # turn data into parseable tensor - using the token IDs instead of raw bytes
    ids = torch.tensor(ids_list, dtype=int_type, device=device)
        # shape (words_in_data * (avg_word_len + 1))\
    
    if demo:
        # Initialize demo text tokens outside the loop to track changes across iterations
        demo_text = (f"This is a test of our custom trained BPE tokenizer on FineWeb data.\n"
                    f"It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly.\n"
                    f"Supercalifragilisticexpialidocious antidisestablishmentarianism!!!")
        demo_words = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    rank = 0
    world_size = 1

    # Now, use our data to figure out which merges we should make
    for j in tqdm(range(256, vocab_size)):
        # find most common pair
        pairs = torch.stack((ids[:-1], ids[1:]), dim=0) # (2, words_in_data * (avg_word_len + 1))
        unique, counts = torch.unique(pairs, return_counts=True, dim=1)
            # shapes (2, words_in_data * (avg_word_len + 1)) and (words_in_data * (avg_word_len + 1))
        valid_mask = torch.all(unique != -1, dim=0)
        unique = unique[:, valid_mask]
        counts = counts[valid_mask]
        counts, sort_idx = torch.sort(counts, descending=True)
        ### single GPU
        #pair_index = sort_idx[0] # shape (1)
        #most_common_pair = unique[:, pair_index].cpu().numpy() # (2)
        ### multi-GPU (simulated for now)
        pairs_idx = sort_idx[rank:rank+world_size] # shape (world_size)
        most_common_pairs_local = unique[:, pairs_idx] # (2, world_size)
        counts_local = counts[pairs_idx] # (world_size)
        most_common_pairs_global = torch.zeros((2,world_size ** 2), dtype=int_type, device=device)
        counts_global = torch.zeros(world_size ** 2, dtype=int_type, device=device)
        most_common_pairs_global[:, rank:rank+world_size] = most_common_pairs_local
        counts_global[rank:rank+world_size] = counts_local
        if j == 256:
            print(most_common_pairs_global)
            print(counts_global)
        # TODO sum across GPUs
        # TODO average any duplicates
        pair_idx = torch.argmax(counts_global) # (1)
        most_common_pair = most_common_pairs_global[:, pair_idx].cpu().numpy() # (2)

        # Map token IDs back to the corresponding byte sequences
        # Using the dictionary in reverse to get the bytes corresponding to these IDs
        most_common_bytes = [None, None]
        for bytes_token, id_token in ranks.items():
            if id_token == most_common_pair[0]:
                most_common_bytes[0] = bytes_token
            if id_token == most_common_pair[1]:
                most_common_bytes[1] = bytes_token
        token_bytes = most_common_bytes[0] + most_common_bytes[1]
        new_token_id = len(ranks)
        # Add the new token!
        ranks[token_bytes] = new_token_id

        # Now merge that most common pair in all the words
        pair_mask = (pairs[0] == most_common_pair[0]) & (pairs[1] == most_common_pair[1]) 
        ids[:-1][pair_mask] = new_token_id
        ids[1:][pair_mask] = -2
        keep_mask = (ids != -2)
        ids = ids[keep_mask]

        if demo:
            # Also apply the same merge to our demo text
            demo_words = slow_merge(demo_words, tuple(most_common_bytes), token_bytes)

            # See the intermediate merges play out!
            if j % 500 == 0 or j in [256, vocab_size - 1]:
                print(f"The most common pair {most_common_pair[0]} + {most_common_pair[1]} "
                        #f"has a count of {counts[pair_index]}\n"
                        f"So we made {token_bytes} our {len(ranks)}th token")
                # Flatten the demo words into a single list of tokens for visualization
                demo_tokens = [token for word in demo_words for token in word]
                visualise_tokens(demo_tokens)
                print("\n")

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


def fetch_fineweb_data(max_chars=2**22, show_stats=True):
    """
    Download a small portion of the FineWeb dataset for tokenizer training
    
    Args:
        max_chars: 
        show_stats: Whether to display statistics about the downloaded data
        
    Returns:
        The concatenated text of all documents
    """
    # Use the 10B version of FineWeb; not edu since the former should have more diverse tokens
    dataset = load_dataset("HuggingFaceFW/fineweb", 
                           name="sample-10BT", 
                           split="train", 
                           streaming=True)
    
    text_data = []
    doc_lengths = []
    tot_len = 0
    for item in dataset:
        if tot_len >= max_chars:
            break
        text_data.append(item["text"])
        doc_lengths.append(len(item["text"]))
        tot_len += len(item["text"])
    
    # Show statistics if requested
    if show_stats:
        print("\nDataset Statistics:")
        print(f"Total documents: {len(text_data)}")
        print(f"Total characters: {sum(doc_lengths):,}")
        print(f"Average document length: {np.mean(doc_lengths):.1f} characters")
        print(f"Median document length: {np.median(doc_lengths):.1f} characters")
        print(f"Shortest document: {min(doc_lengths)} characters")
        print(f"Longest document: {max(doc_lengths):,} characters")
        print(f"Standard deviation: {np.std(doc_lengths):.1f} characters")
    
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
    test_str = "hello world"
    tokens = enc.encode(test_str)
    # Verify encoding-decoding roundtrips correctly
    decoded = enc.decode(tokens)
    assert decoded == test_str, f"Decoding failed: expected '{test_str}' but got '{decoded}'"
    decoded_bytes = enc.decode_bytes(tokens)
    assert decoded_bytes == test_str.encode('utf-8'), \
        f"Bytes decoding failed: expected {test_str.encode('utf-8')} but got {decoded_bytes}"
    
    return enc


def save_tokenizer(enc, filename):
    """Save the tokenizer for later use"""
    tokenizer_data = {"pat_str": enc.pat_str, "mergeable_ranks": enc.mergeable_ranks,}
    f = open(filename + '.pkl', 'wb')
    pickle.dump(tokenizer_data, f)
    print(f"Tokenizer saved to {filename}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("-n", "--samples", type=int, default=2**22, help="Maximum number of text characters to use for training")
    parser.add_argument("-v", "--vocabsize", type=int, default=50256, help="Size of the vocabulary to train (default 50256; one less than GPT2)")
    parser.add_argument("-f", "--savename", type=str, default="custom_tokenizer", help="Filename to save the tokenizer (no extension)")
    parser.add_argument("--demo", action="store_true", default=False, help="Visualize tokenization during training")
    args = parser.parse_args()
    
    # Train the tokenizer
    enc = train_simple_encoding(
        sample_size=args.samples,
        vocab_size=args.vocabsize,
        demo=args.demo
    )
    
    # Save the tokenizer
    save_tokenizer(enc, filename=args.savename)
    