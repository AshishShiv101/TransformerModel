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

## 📌 Features

### 1️⃣ **`BilingualDataset` Class**
A PyTorch `Dataset` that:
- Takes bilingual data in the form:
  ```python
  {'translation': {'en': 'Hello world', 'it': 'Ciao mondo'}}
