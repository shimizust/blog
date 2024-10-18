---
title: "Supercharge LLM Training with Liger Kernel" 
thumbnail: /blog/assets/liger-kernel/thumbnail.gif
authors:
- user: Jason Zhu
- user: Steven Shimizu
---

# Supercharge LLM Training with Liger Kernel 🚀

Training large language models (LLMs) is resource-intensive, requiring significant time and memory. Liger Kernel helps solve this problem by providing highly efficient Triton-based kernels designed to significantly speed up training and reduce memory usage, especially in multi-GPU environments. With integration into Hugging Face's libraries, you can seamlessly optimize your LLM training without sacrificing accuracy in just a few lines of code.

## Benchmarking

| Model            | Original Model | Original Model + Liger | Speedup | VRAM reduction |
|------------------|----------------|------------------------|---------|----------------|
| LLaMA 3.1 8B     | 1x             |                        |         |                |


TODO: Fill in the numbers

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

TODO: Test with TRL flag, validate code
TODO: Add collab links
TODO: Add links to other ways of using Liger Kernel

## Use Case with Medusa Framework: Multiple Decoding Heads

Medusa [[repo](https://github.com/FasterDecoding/Medusa)], [[paper](https://arxiv.org/abs/2401.10774)] is a simple frameowrk for accelerating decoding in transformer models. It adds extra "heads" to the LLM to predict multiple future tokens simultaneously, thus speeding up LLM generation.

When augmenting a model with Medusa, you start with an existing fine-tuned or pre-trained model and add extra heads to the model, keeping the original model weights frozen. The introduction of multiple heads can easily lead to OOM (Out of Memory) issues. However, thanks to the efficient Liger fused CE, which calculates the gradient in place and doesn't materialize the logits, we have observed very effective results. This efficiency opens up more opportunities for multi-token prediction research and development.

TODO: Example code
TODO: Results