# Fine-tuning Llama-3.2-1B on RACE with LoRA

A parameter-efficient fine-tuning pipeline that adapts Meta's Llama-3.2-1B to multiple-choice reading comprehension using LoRA and 4-bit quantization, trained and evaluated on the RACE dataset.

---

## Overview

This project fine-tunes a 1B-parameter causal language model using Low-Rank Adaptation (LoRA) with 4-bit quantization (QLoRA). The model is trained to answer multiple-choice questions from the RACE dataset by generating a single letter (A, B, C, or D). Two inference strategies are compared: direct generation and conditional probability scoring (P(s|c)). Results are assessed both quantitatively (accuracy) and qualitatively (probability distributions per option).

---

## Pipeline

1. Load and filter the RACE dataset
2. Build instruction-style prompts for each example
3. Tokenize with label masking so the model learns only the answer token
4. Load Llama-3.2-1B with 4-bit quantization and apply LoRA adapters
5. Train with the Hugging Face Trainer
6. Evaluate both the base model and the fine-tuned model using two inference methods
7. Perform qualitative comparison of probability distributions between models
8. Save and merge the final LoRA checkpoint

---

## Dataset

- **Source:** [RACE dataset](https://huggingface.co/datasets/ehovy/race) via Hugging Face (`ehovy/race`, split `all`)
- **Filter:** Only articles with fewer than 800 characters are retained
- **Task:** Multiple-choice reading comprehension (4 options, single correct answer: A, B, C, or D)
- **Splits used:** train, validation, test

---

## Requirements

```
transformers
torch
bitsandbytes
peft
datasets
huggingface_hub
numpy
pandas
tqdm
matplotlib
```

Install with:

```bash
pip install --upgrade huggingface_hub transformers -U bitsandbytes peft datasets tqdm
```

> A Hugging Face account with access granted to `meta-llama/Llama-3.2-1B` is required. Authenticate with `huggingface-cli login` or pass your token directly.

---

## Prompt Format

Each example is formatted as a structured instruction prompt:

```
### Article:
<article text>

### Question:
<question>

### Options:
A) ...
B) ...
C) ...
D) ...

### Answer (one letter):
```

During training, all prompt tokens are masked with `-100` in the labels so the cross-entropy loss is computed only over the answer token.

---

## Model and Quantization

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.2-1B` |
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | float16 |
| Framework | BitsAndBytesConfig (bitsandbytes) |

---

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 64 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Bias | none |
| Task type | CAUSAL_LM |

Only the LoRA adapter parameters are trained; the base model weights remain frozen.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size (train) | 4 |
| Batch size (eval) | 8 |
| Gradient accumulation steps | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Precision | fp16 |
| Evaluation strategy | per epoch |
| Max sequence length | 1024 tokens |
| Random seed | 136 |

---

## Inference Methods

Two strategies are used to predict the answer from the model:

**1. Direct generation**  
The prompt is fed to the model and it generates up to 2 new tokens. The first letter of the generated output is taken as the prediction.

**2. Conditional probability scoring — P(s|c)**  
For each option letter (A, B, C, D), the full sequence `prompt + letter` is passed through the model. The average log-probability assigned to the answer token is computed, and the letter with the highest score is selected. This method does not require generation.

---

## Results

| Model | Accuracy (generation) | Accuracy P(s|c) |
|-------|-----------------------|-----------------|
| Base Llama-3.2-1B | 30.2% | — |
| Fine-tuned (LoRA) | 64.6% | — |

Fine-tuning produces a substantial improvement in generation-based accuracy (+34.4 percentage points). However, the probability-based method degrades after fine-tuning: the model collapses probability mass onto a single option (often incorrect), indicating that training on the answer token alone does not optimize the conditional likelihood used by P(s|c).

---

## Qualitative Analysis

The `evaluate_qualitative_comparison` function evaluates a small number of test examples and prints side-by-side probability distributions from the base model and the fine-tuned model for each option. Key observations:

- The base model produces more balanced, better-calibrated probability distributions across options.
- The LoRA model concentrates probability heavily on a single letter, showing high confidence but poor calibration for the P(s|c) method.
- Generation-based inference benefits directly from fine-tuning, while probability-based inference requires a different training objective to be reliable.

---

## Saving the Model

The final LoRA adapter is saved and then merged into the base model weights:

```python
model.save_pretrained("./final_lora_checkpoint")
tokenizer.save_pretrained("./final_lora_checkpoint")

# Merge adapters into base model
merged = PeftModel.from_pretrained(base_model, "./final_lora_checkpoint")
merged = merged.merge_and_unload()
merged.save_pretrained("./final_merged_model")

# Save as a state dict
torch.save(merged.state_dict(), "llama32_race_lora_merged.pt")
```

---

## Project Structure

```
.
├── notebook.ipynb                  # Main notebook with full pipeline
├── final_lora_checkpoint/          # LoRA adapter weights and tokenizer
├── final_merged_model/             # Merged base + LoRA model
├── llama32_race_lora_merged.pt     # State dict of the merged model
└── README.md                       # This file
```

---

## Key Takeaways

- LoRA fine-tuning with answer-token masking is highly effective for improving generation-based accuracy on multiple-choice tasks.
- The training objective (next-token prediction on a single letter) does not align with the P(s|c) evaluation method, which can cause calibration to degrade after fine-tuning.
- 4-bit quantization with LoRA (QLoRA) enables training a 1B-parameter model on consumer-grade hardware with minimal accuracy loss.
