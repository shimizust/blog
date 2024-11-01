---
title: "Supercharge LLM Training with Liger Kernel" 
thumbnail: /blog/assets/liger-kernel/thumbnail.gif
authors:
- user: Steven Shimizu
- user: Jason Zhu
---

# Supercharge LLM Training with Liger Kernel 🚀

Training large language models (LLMs) is GPU resource-intensive, requiring significant time and memory. Liger Kernel helps solve this problem by providing highly efficient Triton-based kernels designed to significantly speed up training and reduce memory usage, especially in multi-GPU environments. With integration into Hugging Face's libraries, you can seamlessly optimize your LLM training without sacrificing accuracy in just a few lines of code.

## Benchmarking

We conducted end-to-end training benchmarks using four NVIDIA A100 GPUs (80 GB each) to fine-tune multiple large language models (LLMs), including LLaMA 3-8B, Qwen2, Gemma, Mistral, and Phi3, on the Alpaca dataset. Sequence length was fixed at 512 tokens and the batch size was varied. We compared the throughput and memory usage of the original Hugging Face model implementations with and without Liger Kernel applied.

The following figure shows a typical result with Gemma2, which has a large vocab size (256K). Applying Liger Kernel not only improves throughput by 12%, but also reduces memory usage by 52% at a batch size of 32. This allows training with smaller GPUs, larger batch sizes, or longer sequence lengths without encountering OOM issues, which can futher improve overall resource and training efficiency.

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/liger-kernel/gemma2_tps_memory_comparison.png" alt="Comparison of throughput and memory for Gemma2 fine-tuning with and without Liger Kernel applied" style="width: 100%;"/>
  <figcaption>Comparison of throughput and memory for Gemma2 fine-tuning with and without Liger Kernel applied.</figcaption>
</figure>

The following table summarizes the throughput (tokens/sec) improvement and GPU memory reduction when applying Liger. results for other models (for specific batch sizes). The full benchmarking details and results are discussed in https://arxiv.org/abs/2410.10989. Example benchmarking scripts can be found here: 

| Model            | Throughput (tokens/sec) Improvement with Liger Kernel | VRAM Reduction with Liger Kernel |
|------------------|-------------------------------------------------------|----------------------------------|
| LLaMA 3 8B       | +43%                                                  | -55%                             |
| Qwen2            | +26%                                                  | -57%                             |  
| Gemma2           | +12%                                                  | -52%                             |
| Mistral          | +27%                                                  | -21%                             |
| Phi3             | +17%                                                  | -13%                             |


## Usage

Liger Kernel contains PyTorch `nn.Module` implementations built on Triton kernels, and these are compliant with the API of several common LLM layers present in `transformers` modeling code. For example, layers such as RMSNorm, RoPE, SwiGLU, and CrossEntropy have corresponding Liger Kernel implementations that can be patched to significantly boost throughput and reduce memory usage during training.

While many of the models in the `transformers` code base reuse implementations from some base models (e.g. `LLaMA`), there may be slight differences in the layers. Thus, we provide model-specific monkey patch APIs that have been thoroughly tested for accuracy and performance. At the time of writing, the following models are supported:

- LLaMA 2 & 3
- LLaMA 3.2-Vision
- Mistral
- Mixtral
- Gemma1
- Gemma2
- Qwen2 & Qwen2.5
- Qwen2-VL
- Phi3 & Phi3.5

### AutoLigerKernelForCausalLM

If you are using a CausalLM type of model, the `AutoLigerKernelForCausalLM` class automatically checks if Liger Kernel supports the model and applies the appropriate monkey patch API.

```python
from liger_kernel.transformers import AutoLigerKernelForCausalLM

model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
```

### HuggingFace Trainer Integration

As an alternative to patching the model before training, you can also provide a `use_liger` flag to the `transformers`' `Trainer` class to automatically apply the Liger Kernel optimizations.

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    trust_remote_code=True,
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./llama_finetuning",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        use_liger=True,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)
trainer.train()
```

## Use Case with Medusa Framework: Multiple Decoding Heads

Medusa [[repo](https://github.com/FasterDecoding/Medusa)], [[paper](https://arxiv.org/abs/2401.10774)] is a simple frameowrk for accelerating decoding in transformer models. It adds extra "heads" to the LLM to predict multiple future tokens simultaneously, thus speeding up LLM generation.

When augmenting a model with Medusa, you start with an existing fine-tuned or pre-trained model and add extra heads to the model, keeping the original model weights frozen. The introduction of multiple heads can easily lead to OOM (Out of Memory) issues. However, thanks to the efficient Liger fused CE, which calculates the gradient in place and doesn't materialize the logits, we have observed very effective results. This efficiency opens up more opportunities for multi-token prediction research and development.

TODO: Example code
TODO: Results