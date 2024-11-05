---
title: "Supercharge LLM Training with Liger Kernel" 
thumbnail: /blog/assets/liger-kernel/thumbnail.gif
authors:
- user: Byron Hsu
- user: Steven Shimizu
---

# Supercharge LLM Training with Liger Kernel 🚀

Training large language models (LLMs) is GPU resource-intensive, requiring significant time and memory. Liger Kernel helps solve this problem by providing highly efficient Triton-based kernels designed to significantly speed up training and reduce memory usage, especially in multi-GPU environments. With integration into Hugging Face's libraries, you can seamlessly optimize your LLM training without sacrificing accuracy in just a few lines of code.

## Benchmarking

We conducted end-to-end training benchmarks using four NVIDIA A100 GPUs (80 GB each) to fine-tune multiple large language models (LLMs), including LLaMA 3-8B, Qwen2, Gemma, Mistral, and Phi3, on the Alpaca dataset. Sequence length was fixed at 512 tokens and the batch size was varied. We compared the throughput and memory usage of the original Hugging Face model implementations with and without Liger Kernel applied.

Our benchmarking illustrates Liger's impact across various batch sizes and model architectures. Below is a typical result with Gemma2, a model with a large vocab size (256K). Applying Liger Kernel not only improves throughput by 12%, but also reduces memory usage by 52% at a batch size of 32. This allows training with smaller GPUs, larger batch sizes, or longer sequence lengths without encountering OOM issues, which can futher improve overall resource and training efficiency.

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

# Train the model as usual
```

### HuggingFace Trainer Integration

As an alternative to patching the model before training, you can also set the `use_liger_kernel` flag in the `transformers`' `Trainer` class to automatically apply the Liger Kernel optimizations. This requires `transformers` version >=4.45.0.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load Llama-3.1 model, which is supported by Liger Kernel
model_path = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)


dataset = load_dataset("stanfordnlp/imdb", split="train")

def tokenize_function(examples):
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="/tmp",
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    num_train_epochs=1,
    use_liger_kernel=True  # Enable Liger optimization for training
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Train the model
trainer.train()
```

### TRL SFTTrainer Integration

The `SFTTrainer` from the `trl` library simplifies the process of fine-tuning large language models. By simply passing the `use_liger` flag to the `SFTTrainer`, it will automatically apply the Liger Kernel optimizations on the specified model.

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("stanfordnlp/imdb", split="train")

training_args = SFTConfig(
    max_seq_length=512,
    output_dir="/tmp",
    use_liger_kernel=True,
)
trainer = SFTTrainer(
    "meta-llama/Llama-3.1-8B",
    train_dataset=dataset,
    args=training_args,
)
trainer.train()
```

### Advanced Integration Options

If you’re working with new model architectures or want finer control, you can directly use the optimized `nn.Module` layers or Triton ops provided by Liger Kernel to fit your specific needs. For details on patching layers and customizing Triton ops for advanced integration, check out the [GitHub repository docs](https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#3-compose-your-own-model).

## Conclusion

Liger Kernel is a powerful solution for making large language model training more efficient. With Triton-based kernels optimized for common LLM layers, it boosts throughput and reduces memory usage, enabling faster training and better resource utilization. Integrated seamlessly with Hugging Face's libraries, Liger Kernel optimizations can be applied in just a few lines of code, helping to accelerate LLM development.

Whether you're exploring Liger Kernel for the first time or ready to contribute new kernels, bug fixes or insights, your involvement helps push model efficiency forward for everyone. Join the Liger community on [GitHub](https://github.com/linkedin/Liger-Kernel)!
