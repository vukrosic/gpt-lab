import tiktoken
import json

cl100k_base = tiktoken.get_encoding("cl100k_base")
print(cl100k_base._pat_str)
print(type(cl100k_base._mergeable_ranks))
for i, (key, val) in enumerate(cl100k_base._mergeable_ranks.items()):
    if i > 5 and i < len(cl100k_base._mergeable_ranks) - 5:
        continue
    print(i, key, val)
    print(type(key))
# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings


with open('custom_tokenizer.json', 'r') as f:
    tokenizer_config = json.load(f)
# Initialize the tokenizer with the loaded configuration
# Convert back to bytes objects
mergeable_ranks = {
    #bytes(k): v for k, v in tokenizer_config["mergeable_ranks"].items()
    k.encode('utf-8'): v for k, v in tokenizer_config["mergeable_ranks"].items()
}

for i, (key, val) in enumerate(mergeable_ranks.items()):
    if i > 5 and i < len(mergeable_ranks) - 5:
        continue
    print(i, key, val)
    print(type(key))

enc1 = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)


enc2 = tiktoken.Encoding(
    name="custom",
    pat_str=tokenizer_config['pat_str'],
    mergeable_ranks=mergeable_ranks,
    special_tokens={
        "<|endoftext|>": len(mergeable_ranks),
    }
)