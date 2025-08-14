# 🌍 Bilingual Dataset Loader for Transformer-based Translation

This repository contains a **PyTorch Dataset** and helper utilities for preparing bilingual datasets for **sequence-to-sequence Transformer models** (such as the "Attention Is All You Need" architecture).  
It is designed for **machine translation** tasks and covers **tokenization, padding, masking, and config management**.

---

## 📖 Overview

When training a Transformer for translation (e.g., English → Italian), the model needs:
1. **Numericalized text** (tokens → IDs)
2. **Uniform sequence length** (padding or truncation)
3. **Special tokens** (`[SOS]`, `[EOS]`, `[PAD]`)
4. **Masks** to control which positions the model attends to  
   - **Padding masks** so the model ignores `[PAD]` tokens.
   - **Causal masks** so the decoder can't "see the future".

This project implements:
- A `BilingualDataset` for preprocessing and batching bilingual text data.
- Mask creation logic for both encoder and decoder.
- Config management for training parameters.
- Utility functions for saving/loading model weights.

---

## 🧠 Inspiration

This implementation is heavily inspired by the **Transformer architecture** introduced in the paper:  
> *Vaswani, A., et al. (2017). "Attention Is All You Need."*  
> [🔗 Read the paper (arXiv)](https://arxiv.org/abs/1706.03762)

Key elements adopted from the paper:
- **Encoder–Decoder Structure**  
  Our `BilingualDataset` prepares `encoder_input` and `decoder_input` exactly as described in the Transformer model for sequence-to-sequence tasks.
- **Attention Masks**  
  Implements **causal masks** in the decoder to prevent attention to future tokens (as in the paper’s Figure 1) and **padding masks** to handle variable-length sentences.
- **Position Handling**  
  The fixed `seq_len` design ensures compatibility with **positional encoding** (Section 3.5 of the paper).
- **Teacher Forcing Setup**  
  `decoder_input` is the target sequence shifted right by one token, following the paper’s training approach.

By following the original Transformer preprocessing principles, this dataset loader ensures that the model receives inputs in the exact format required for optimal attention computation.

---

## 📌 Features

### 1️⃣ **`BilingualDataset` Class**
A PyTorch `Dataset` that:
- Takes bilingual data in the form:
  ```python
  {'translation': {'en': 'Hello world', 'it': 'Ciao mondo'}}
