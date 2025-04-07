import os
import sys
from huggingface_hub import hf_hub_download

# Dataset configurations
DATASETS = {
    'fineweb10B': {
        'repo_id': 'kjj0/fineweb10B-gpt2',
        'dir_name': 'fineweb10B',
        'file_prefix': 'fineweb',
        'total_chunks': 103
    },
    'fineweb100B': {
        'repo_id': 'kjj0/fineweb100B-gpt2',
        'dir_name': 'fineweb100B',
        'file_prefix': 'fineweb',
        'total_chunks': 1030
    },
    'finewebedu10B': {
        'repo_id': 'kjj0/finewebedu10B-gpt2',
        'dir_name': 'finewebedu10B',
        'file_prefix': 'finewebedu',
        'total_chunks': 99
    }
}

def print_help():
    """Prints the help message explaining the script usage."""
    help_message = """
    Usage: python gpt2_download.py [dataset] [num_chunks]

    Options:
      dataset     The name of the dataset to download. Available options are:
                  - fineweb10B
                  - fineweb100B
                  - finewebedu10B (default)
      num_chunks  The number of chunks to download (100 million tokens per chunk).
                  If not specified, the full dataset will be downloaded.

    Examples:
      python gpt2_download.py                # Downloads full finewebedu10B
      python gpt2_download.py 50             # Downloads 50 chunks of finewebedu10B
      python gpt2_download.py fineweb10B     # Downloads full fineweb10B
      python gpt2_download.py fineweb100B 100 # Downloads 100 chunks of fineweb100B
    """
    print(help_message)

def download_dataset(dataset_name, num_chunks=None):
    """Download the specified dataset with optional chunk count limit"""
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found. Available options: {', '.join(DATASETS.keys())}")
        return
    
    config = DATASETS[dataset_name]
    
    def get(fname):
        local_dir = os.path.join(os.path.dirname(__file__), config['dir_name'])
        if not os.path.exists(os.path.join(local_dir, fname)):
            hf_hub_download(repo_id=config['repo_id'], filename=fname,
                            repo_type="dataset", local_dir=local_dir)
    
    # Download validation file
    get(f"{config['file_prefix']}_val_{0:06d}.bin")
    
    # Determine number of chunks to download
    total_chunks = num_chunks if num_chunks is not None else config['total_chunks']
    print(f"Downloading {dataset_name} dataset ({total_chunks} chunks)...")
    
    # Download training files
    for i in range(1, total_chunks+1):
        get(f"{config['file_prefix']}_train_{i:06d}.bin")
    
    print(f"Download of {dataset_name} completed.")

if __name__ == "__main__":
    # Check for help argument
    if len(sys.argv) > 1 and sys.argv[1] in ('--help', '-h'):
        print_help()
        sys.exit(0)
    
    # Default to finewebedu10B
    dataset = 'finewebedu10B'
    chunks = None
    
    # Parse command line arguments
    if len(sys.argv) >= 2:
        # Check if first argument is a dataset name
        if sys.argv[1] in DATASETS:
            dataset = sys.argv[1]
            if len(sys.argv) >= 3:
                chunks = int(sys.argv[2])
        else:
            # First argument must be chunk count for default dataset
            chunks = int(sys.argv[1])
    
    download_dataset(dataset, chunks)