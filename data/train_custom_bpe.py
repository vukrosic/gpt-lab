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

import torch
device = "cuda" if torch.cuda.is_available() \
    else 'mps' if torch.backends.mps.is_available() \
    else "cpu"

class SimpleBytePairEncoding:
    def __init__(self, *, pat_str: str, mergeable_ranks: dict[bytes, int]) -> None:
        """Creates an Encoding object."""
        # A regex pattern string that is used to split the input text
        self.pat_str = pat_str
        # A dictionary mapping token bytes to their ranks. The ranks correspond to merge priority
        self.mergeable_ranks = mergeable_ranks

        self._decoder = {token: token_bytes for token_bytes, token in mergeable_ranks.items()}
        self._pat = regex.compile(pat_str)

    def encode(self, text: str, visualise: str | None = None) -> list[int]:
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
            word_tokens = bpe_encode(self.mergeable_ranks, word_bytes, visualise=visualise)
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
    def train(training_data: str, vocab_size: int, pat_str: str, visualise: bool = False):
        """Train a BPE tokeniser on some data!"""
        mergeable_ranks = bpe_train(data=training_data, vocab_size=vocab_size-1, pat_str=pat_str, visualise=visualise)
        return SimpleBytePairEncoding(pat_str=pat_str, mergeable_ranks=mergeable_ranks)

    @staticmethod
    def from_tiktoken(encoding):
        if isinstance(encoding, str):
            encoding = tiktoken.get_encoding(encoding)
        return SimpleBytePairEncoding(
            pat_str=encoding._pat_str, mergeable_ranks=encoding._mergeable_ranks
        )


def bpe_encode(
    mergeable_ranks: dict[bytes, int], input: bytes, visualise: str | None = None
) -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        # See the intermediate merges play out!
        if visualise: visualise_tokens(parts)

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

def merge(words, most_common_pair, token_bytes):
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
    data: str, vocab_size: int, pat_str: str, visualise: bool = False
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

    # Initialize demo text tokens outside the loop to track changes across iterations
    demo_text = "This is a test of our custom trained BPE tokenizer on FineWeb data. It should handle punctuation, numbers (like 42 and 3.14159), and special characters ($#@!) properly."
    demo_words = [[bytes([b]) for b in word.encode("utf-8")] for word in regex.findall(pat_str, demo_text)]

    # Now, use our data to figure out which merges we should make
    for j in tqdm(range(257, vocab_size)):
        # turn data into parseable tensor
        m = max([len(word) for word in words])
        words_tensor = [torch.cat(
            torch.tensor(word, dtype=int_type, device=device),
            -1 * torch.ones(m - len(word) + 1, dtype=int_type, device=device),
            dim = 0) for word in words]
        words_tensor = torch.cat(tuple(word.unsqueeze(0) for word in words_tensor), dim=0)

        # find most common pair
        pairs = torch.stack((words_tensor[:, :-1], words_tensor[:, 1:]), dim=0)
        pairs = pairs.reshape(pairs.shape[0], -1)
        unique, counts = torch.unique(pairs, sorted=False, return_counts=True, dim=1)
        mask = torch.all(unique != -1, dim=0)
        valid_counts = torch.where(mask, counts, torch.tensor(0, dtype=counts.dtype))
        pair_index = torch.argmax(valid_counts)
        most_common_pair, count = unique[:, pair_index], counts[pair_index]

        # add new byte pair encoding to ranks
        token_bytes = most_common_pair[0].item() + most_common_pair[1].item()
        token = len(ranks)
        # Add the new token!
        ranks[token_bytes] = token

        # Now merge that most common pair in all the words. That is, update our training data
        # to reflect our decision to make that pair into a new token.
        words = merge(words, most_common_pair, token_bytes)
        # Also apply the same merge to our demo text
        demo_words = merge(demo_words, most_common_pair, token_bytes)

        # See the intermediate merges play out!
        if visualise and (j % 50 == 0 or j in [257, vocab_size - 1]):
            print(f"The current most common pair is {most_common_pair[0]} + {most_common_pair[1]}")
            print(f"So we made {token_bytes} our {len(ranks)}th token")
            print("Now our demo text looks like:")
            # Flatten the demo words into a single list of tokens for visualization
            demo_tokens = [token for word in demo_words for token in word]
            visualise_tokens(demo_tokens)
            print("\n")

    return ranks


def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
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


def fetch_fineweb_data(max_samples=100, download_dir="data/fineweb_temp", show_stats=True):
    """
    Download a small portion of the FineWeb dataset for tokenizer training
    
    Args:
        max_samples: Number of documents to download
        download_dir: Directory to store downloaded data (will be created if needed)
        show_stats: Whether to display statistics about the downloaded data
        
    Returns:
        The concatenated text of all documents
    """
    # Create download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)
    
    # Check if data already exists (using cached data file as indicator)
    # NOTE DOES NOT CHECK TO SEE IF CACHED DATA IS SIZE REQUESTED BY INPUT ARGS
    data_cache_file = os.path.join(download_dir, "cached_data.txt")
    if os.path.exists(data_cache_file):
        print(f"Found existing data in {download_dir}. Using cached data.")
        with open(data_cache_file, "r") as f:
            return f.read()
    
    print(f"Downloading {max_samples} samples from the FineWeb dataset to {download_dir}...")
    # Use the 10B version of FineWeb as it's smaller
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", 
                          streaming=True)#, cache_dir=download_dir)
    
    # Take only a tiny portion for training
    text_data = []
    doc_lengths = []
    
    for i, item in enumerate(tqdm(dataset, total=max_samples, desc="Downloading documents")):
        if i >= max_samples:
            break
        text_data.append(item["text"])
        doc_lengths.append(len(item["text"]))
    
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
    joined_data = "\n".join(text_data)
    
    # Cache the data for future use
    with open(data_cache_file, "w") as f:
        f.write(joined_data)
    
    return joined_data


def cleanup_download_dir(download_dir="data/fineweb_temp"):
    """Remove temporary download directory to clean up disk space"""
    if os.path.exists(download_dir):
        print(f"Cleaning up temporary download directory: {download_dir}")
        shutil.rmtree(download_dir)


def train_simple_encoding(sample_size=100, vocab_size=600, download_dir="data/fineweb_temp",
                          show_stats: bool =True, clean_up: bool =True, visualise: bool = False):
    """
    Train a custom BPE tokenizer using FineWeb data.
    
    Args:
        sample_size: Number of FineWeb samples to use
        vocab_size: Size of the vocabulary to train
        download_dir: Directory to store downloaded data
        show_stats: Whether to display statistics about the downloaded data
        clean_up: Whether to delete the download directory after training
        visualise: Visualization mode for BPE training process ("color", "simple", or None)
    
    Returns:
        The trained tokenizer
    """
    gpt2_pattern = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    #gpt4_pattern = (
        #r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
    #)
    
    print(f"Training tokenizer on {sample_size} FineWeb samples with vocab_size={vocab_size}")
    data = fetch_fineweb_data(max_samples=sample_size, download_dir=download_dir, show_stats=show_stats)

    # Train the tokenizer with the specified vocabulary size (one less to account for the special token)
    enc = SimpleBytePairEncoding.train(data, vocab_size=vocab_size, pat_str=gpt2_pattern, visualise=visualise)
    
    # Clean up download directory if requested
    if clean_up:
        cleanup_download_dir(download_dir)

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


def save_tokenizer(enc, filename="custom_tokenizer.json"):
    """Save the tokenizer for later use"""
    import json
    
    # Convert bytes keys to strings for JSON serialization
    mergeable_ranks = {str(k.hex()): v for k, v in enc.mergeable_ranks.items()}
    
    tokenizer_data = {
        "pat_str": enc.pat_str,
        "mergeable_ranks": mergeable_ranks,
    }
    
    with open(filename, "w") as f:
        json.dump(tokenizer_data, f)
    
    print(f"Tokenizer saved to {filename}")


def load_tokenizer(filename="tokenizer.json"):
    """Load a previously saved tokenizer"""
    import json
    
    with open(filename, "r") as f:
        data = json.load(f)
    
    # Convert string keys back to bytes
    mergeable_ranks = {bytes.fromhex(k): v for k, v in data["mergeable_ranks"].items()}
    
    # Create tokenizer
    tokenizer = SimpleBytePairEncoding(pat_str=data["pat_str"], mergeable_ranks=mergeable_ranks)
    
    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom BPE tokenizer")
    parser.add_argument("--samples", type=int, default=100, help="Number of FineWeb samples to use for training")
    parser.add_argument("--vocab-size", type=int, default=600, help="Size of the vocabulary to train")
    parser.add_argument("--save", type=str, default="tokenizer.json", help="Filename to save the tokenizer")
    parser.add_argument("--download-dir", type=str, default="fineweb_temp", help="Directory for temporary download files")
    parser.add_argument("--delete_dataset", action="store_true", help="Delete downloaded data after training")
    parser.add_argument("--visualise", action="store_true", help="Visualization mode during training")
    args = parser.parse_args()
    
    # Set visualise to None if 'none' is selected
    visualise = None if args.visualise == "none" else args.visualise
    
    # Train the tokenizer
    enc = train_simple_encoding(
        sample_size=args.samples,
        vocab_size=args.vocab_size,
        download_dir=args.download_dir,
        clean_up=args.delete_dataset,  # Now only delete if --delete flag is provided
        visualise=visualise
    )
    
    # Save the tokenizer
    save_tokenizer(enc, filename=args.save)
    