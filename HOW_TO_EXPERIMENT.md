# How to Conduct Scientifically Robust Experiments

This document outlines best practices for designing and running experiments using the provided codebase (`train_gpt.py`, `download_fineweb.py`, `train_tokenizer.py`) to ensure results are reliable, reproducible, and scientifically valid.

## 1. Defining the Experiment

Before writing any code, clearly define:

*   **Hypothesis:** What specific question are you trying to answer? What effect do you expect your change to have? (e.g., "Hypothesis: Replacing the GELU activation in the MLP with Squared ReLU will improve validation loss for a fixed parameter count.")
*   **Independent Variable:** What single change are you introducing? (e.g., the MLP activation function).
*   **Dependent Variables:** What metrics will you measure to evaluate the change? (e.g., validation loss, training throughput, HellaSwag accuracy).
*   **Control Variables:** What aspects will you keep constant to ensure a fair comparison? (See Section 3).

## 2. Measuring the Impact of Changes

Quantifying the effect of your changes is crucial. Here's how to measure key aspects:

*   **Parameter Count:**
    *   Use the `model.get_num_params()` method in `train_gpt.py`.
    *   Log this value at the start of training. Compare the parameter count of your modified model to the baseline.
*   **Compute:**
    *   **Training Time:** The most direct measure available in this codebase. The logging in `train_gpt.py` records `cumulative_time_ms` and `step_avg_ms`. Compare the total training time or time per step for a fixed number of `train_steps`.
    *   **FLOPs (Floating Point Operations):** Estimating FLOPs precisely can be complex.
        *   For transformer models, FLOPs are dominated by matrix multiplications. Major changes (embedding dim, num layers, num heads, sequence length, mlp ratio) significantly impact FLOPs. Minor changes (e.g., activation function, normalization type) have a smaller effect.
        *   You can roughly estimate the change in FLOPs by considering how your edit affects the dimensions of the main matrix multiplications (QKV projection, attention output projection, MLP layers).
        *   A rough approximation for a standard transformer forward pass is \( \approx 2 \times \text{num_params} \times \text{seq_len} \) FLOPs per token. Use this as a guideline, but focus on empirical measures like training time when possible.
*   **Memory Usage:**
    *   `train_gpt.py` logs peak GPU memory allocated and reserved using `torch.cuda.max_memory_allocated()` and `torch.cuda.max_memory_reserved()`.
    *   Compare these values between runs. Note that memory usage can depend on `batch_size` (`train_seq_len` * `world_size` * `grad_acc_steps`), model architecture, and activation checkpointing (not used here).
*   **Data Throughput:**
    *   Monitor tokens per second, which can be derived from `step_avg_ms`, `train_seq_len`, `world_size`, and `grad_acc_steps`. `Tokens/sec = (train_seq_len * world_size * grad_acc_steps) / (step_avg_ms / 1000)`.
*   **Tokenization Efficiency:** (Relevant if modifying `train_tokenizer.py`)
    *   Train two tokenizers (baseline vs. modified) on the *same* data sample.
    *   Encode a fixed, large text corpus (e.g., a validation shard from `download_fineweb.py`) with both tokenizers.
    *   Compare the total number of tokens produced. Fewer tokens for the same text indicate better compression.
    *   Compare the resulting vocabulary sizes.

## 3. Controlling Variables: Apples-to-Apples Comparisons

To isolate the effect of your change, you MUST keep other factors constant. Deciding *what* to hold constant depends on your hypothesis:

*   **Fixed Parameter Count:**
    *   **When:** Comparing architectural changes (e.g., different attention mechanisms, layer types) where you want to see if one is inherently "better" given the same number of parameters.
    *   **How:** If your change adds parameters (e.g., a new small layer), you must reduce parameters elsewhere (e.g., slightly decrease `model_dim` or `mlp_ratio`) to match the baseline total. Use `model.get_num_params()` to verify.
*   **Fixed Compute Budget (Training Time / FLOPs):**
    *   **When:** Comparing changes that might trade parameters for speed (or vice-versa), or when evaluating practical efficiency improvements. Often the most relevant comparison for real-world application.
    *   **How:** Run both baseline and experiment for the same duration *or* the same number of `train_steps`. If one model trains significantly faster per step, it will complete more steps (and see more data) in a fixed time budget. If comparing architectures with different FLOPs/step, you might need to adjust `train_steps` so the *total* estimated FLOPs are similar. Focus on comparing validation loss curves against training time or steps.
*   **Fixed Memory Usage:**
    *   **When:** Evaluating changes designed to reduce memory footprint (e.g., different activation functions, custom kernels, quantization like the FP8 linear layer).
    *   **How:** Monitor peak memory usage. If your change reduces memory, you might be able to *increase* batch size (`train_seq_len` or `grad_acc_steps`) to utilize the saved memory, potentially improving throughput (compare this against the baseline at its maximum possible batch size).
*   **Fixed Hyperparameters:**
    *   **Always:** Unless the hyperparameter itself is the independent variable, keep learning rate schedules, optimizer settings (betas, epsilon), `batch_size`, `train_seq_len`, `train_steps`, `cooldown_frac`, etc., identical between runs.
*   **Fixed Data:**
    *   **Always:** Use the exact same training and validation datasets (`train_files`, `val_files`, `val_tokens`). Ensure the data loading and preprocessing (`distributed_data_generator`) are identical.
*   **Fixed Tokenizer:**
    *   **Always:** Unless experimenting *with* the tokenizer, use the exact same `tokenizer` file for all runs.

**Trade-offs:** You often can't hold everything constant. For example, adding a skip connection might slightly increase parameters and FLOPs. Decide what comparison is most meaningful for your hypothesis. Is it better *at the same size*? Is it better *given the same training time*? Be explicit about what you held constant and why.

## 4. Ablation Studies

Ablation studies involve systematically removing components of a system to understand their contribution.

*   **One Change at a Time:** Start with a complex model or technique (e.g., the current `train_gpt.py` model) and create variants where *one* feature is removed or replaced with a simpler alternative.
    *   Example Ablations for `train_gpt.py`:
        *   Remove QK Norm: Disable `norm(q), norm(k)` in `CausalSelfAttention`.
        *   Replace Squared ReLU: Use standard `F.relu(x)` or `F.gelu(x)` in `MLP`.
        *   Remove RoPE: Pass identity function instead of `self.rotary(q), self.rotary(k)`.
        *   Remove Token Value Embeddings: Set `ve = None` in `GPT.forward`.
        *   Remove MLP Zero Init: Remove `self.c_proj.weight.detach().zero_()` in `MLP`.
        *   Remove U-Net Skips: Remove the skip connection logic in `GPT.forward`.
        *   Remove Attention Skip Layer: Don't skip attention in `Block` (remove `skip_attn` logic).
*   **Compare to Baseline:** Always compare the ablated model against the full model (your baseline for the ablation).
*   **Control Variables:** Apply the principles from Section 3. Decide whether to maintain parameter count, compute, etc., during the ablation. Often, maintaining compute (training steps/time) is most informative for ablations.
*   **Document:** Clearly state what was removed/changed in each ablation run.

## 5. Reproducibility and Logging

*   **Code Versioning:** Use Git. Commit your changes before each experiment run. Record the commit hash used for each run.
*   **Environment:** Record package versions (`requirements.txt`), PyTorch version, CUDA version, and hardware used (GPU type). The logging in `train_gpt.py` already captures some of this.
*   **Experiment Tracking:**
    *   Leverage the built-in logging (`experiments/` directory, `training_log.txt`, `metrics.csv`).
    *   Consider using tools like Weights & Biases or MLflow for more sophisticated tracking, especially for many runs.
    *   Ensure the `model_name` and command-line arguments used are logged (the script attempts to do this).
*   **Random Seeds:** While not explicitly set in the current `train_gpt.py`, for ultimate reproducibility, set seeds for Python's `random`, `numpy`, and `torch` at the beginning of your script. Note that some CUDA operations can still be non-deterministic. Run experiments multiple times with different seeds if high confidence is needed.
*   **Statistical Significance:** A single run might be noisy. If computationally feasible, run each configuration (baseline and experiment) multiple times (e.g., 3-5 times) with different random seeds. Report mean and standard deviation for key metrics. Use statistical tests (like t-tests) if appropriate to determine if differences are significant.

## 6. Interpreting Results

*   **Look at Curves:** Don't just compare final numbers. Analyze the validation loss curves (`metrics.csv`) over training time/steps. Does the change lead to faster convergence? A lower final loss? More stable training?
*   **Consider Trade-offs:** Did the change improve validation loss but significantly slow down training? Is the improvement worth the added complexity or compute?
*   **Be Critical:** Was the experiment truly fair? Could other factors explain the difference? Acknowledge limitations.
*   **Report Clearly:** Document your setup, hypothesis, methodology, results (including metrics, plots, and what was held constant), and conclusions clearly. Share your code and experiment logs.

By following these practices, you can increase the confidence in your results and contribute meaningfully to the understanding of these models and techniques.