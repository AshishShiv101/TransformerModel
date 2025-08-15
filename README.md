# Bilingual Translation Dataset & Configuration for Transformer Models

Natural Language Processing (NLP) has been revolutionized by the Transformer architecture, introduced in the landmark paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al., 2017.  
The paper demonstrated that attention-based mechanisms could replace recurrent and convolutional structures, enabling faster training, better parallelization, and state-of-the-art performance in sequence-to-sequence tasks like machine translation.

This project implements a **custom bilingual dataset handler** and **training configuration setup** for building Transformer-based models for neural machine translation (NMT).  
It is designed to be flexible, reproducible, and inspired by the key ideas in the original Transformer paper.

---

## 📜 Key Concepts from *Attention Is All You Need*
- **Self-Attention Mechanism** – The model learns relationships between all words in a sequence, regardless of distance.
- **Encoder-Decoder Structure** – Encodes the source language into a representation, then decodes it into the target language.
- **Positional Encoding** – Adds order information to token embeddings since attention is order-agnostic.
- **Causal Masking** – Prevents the model from "peeking ahead" when predicting the next word during decoding.
- **Parallelism** – Unlike RNNs, the entire sequence is processed simultaneously.

---

## ✨ Features

### 1. **Bilingual Dataset Class**
A `torch.utils.data.Dataset` subclass (`BilingualDataset`) that:
- Loads parallel text pairs (`src_lang` → `tgt_lang`).
- Uses separate tokenizers for source and target languages.
- Adds special tokens:  
  - **[SOS]** – Start of sequence  
  - **[EOS]** – End of sequence  
  - **[PAD]** – Padding for equal sequence length  
- Ensures fixed-length sequences for both encoder and decoder inputs.
- Automatically creates **masks** for attention layers:
  - **Encoder Mask** – Ignores padding tokens in the encoder.
  - **Decoder Mask** – Combines padding mask with **causal mask** to block future tokens.

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd09109e-24c6-4ed7-8bc8-15df9ef0a872" alt="Encoder-Decoder Attention" width="75%">
</p>

---

### 2. **Cross Attention Visualization**
<p align="center">
  <img src="https://github.com/user-attachments/assets/8344a080-93d2-43c5-8377-f2275d3011b4" alt="Cross Attention" width="75%">
</p>

---

### 3. **Causal Mask Function**
Implements a **look-ahead mask** for the decoder:

```python
def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
