import tiktoken
import json

cl100k_base = tiktoken.get_encoding("cl100k_base")
print(cl100k_base._pat_str)
print(type(cl100k_base._mergeable_ranks))
for i, (key, val) in enumerate(zip(cl100k_base._mergeable_ranks.keys(), cl100k_base._mergeable_ranks.values())):
    if i > 5 and i < len(cl100k_base._mergeable_ranks) - 5:
        continue
    print(i, key, val)
# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings


with open('custom_tokenizer.json', 'r') as f:
    tokenizer_config = json.load(f)
# Initialize the tokenizer with the loaded configuration
print(tokenizer_config['pat_str'])
print(type(tokenizer_config['mergeable_ranks']))
for i, (key, val) in enumerate(zip(tokenizer_config['mergeable_ranks'].keys(), tokenizer_config['mergeable_ranks'].values())):
    if i > 5 and i < len(tokenizer_config['mergeable_ranks']) - 5:
        continue
    print(i, key, val)

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
    mergeable_ranks=tokenizer_config['mergeable_ranks'],
    special_tokens={
        "<|endoftext|>": len(tokenizer_config['mergeable_ranks']),
    }
)