# Multimodal Vision-Language Pretraining (CLIP) from Scratch in PyTorch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A clean, from-scratch PyTorch implementation of OpenAI's **CLIP (Contrastive Language-Image Pretraining)** architecture. This repository contains the standalone Vision Transformer (ViT) image encoder, causal masked text encoder with QuickGELU activations, MS COCO caption dataset pipeline, and symmetric InfoNCE contrastive training loop with automatic mixed-precision (AMP).

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Architecture Details](#architecture-details)
  - [1. Vision Transformer (`image_encoder.py`)](#1-vision-transformer-image_encoderpy)
  - [2. Causal Text Encoder (`text_encoder.py`)](#2-causal-text-encoder-text_encoderpy)
  - [3. Contrastive Objective & Loss (`train.py`)](#3-contrastive-objective--loss-trainpy)
  - [4. Dataset & Preprocessing (`coco_dataset.py`)](#4-dataset--preprocessing-coco_datasetpy)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Training Pipeline](#training-pipeline)
  - [Dataset Preparation](#dataset-preparation)
  - [Running the Training Script](#running-the-training-script)
- [Testing & Verification](#testing--verification)

---

## Repository Structure

The project has a modular, flat structure containing pure PyTorch module implementations and the training harness:

```
CLIP-Implementation/
├── coco_dataset.py      # CustomCLIPDataset wrapping torchvision.datasets.CocoCaptions
├── image_encoder.py     # ViT backbone built with einops, MHSA, and MLP heads
├── text_encoder.py      # Causal masked Transformer with QuickGELU and [EOS] token pooling
├── train.py             # CLIPModel wrapper, symmetric cross-entropy loss, and AMP training loop
└── README.md
```

---

## Architecture Details

### 1. Vision Transformer (`image_encoder.py`)
- **Patch Extraction & Projection:** Uses `einops.layers.torch.Rearrange` to slice input images $x \in \mathbb{R}^{B \times C \times H \times W}$ into non-overlapping $16 \times 16$ patches, projected linearly to latent dimension `dim`.
- **Positional & Class Embeddings:** Prepends a learnable `[CLS]` token and adds 1D learnable positional embeddings across all patch tokens.
- **Encoder Blocks:** Implements `TransformerBlock` containing `nn.MultiheadAttention`, Layer Normalization, and a 2-layer `FeedForward` MLP with GELU activations.
- **Pooling & Projection:** Supports both `cls` token extraction and mean pooling, followed by a LayerNorm and linear projection `mlp_head`.

### 2. Causal Text Encoder (`text_encoder.py`)
- **Token & Position Embeddings:** Embeds tokenized sequences (default context length: 77, vocabulary size: 49,408) and adds learnable positional encodings.
- **Causal Masking:** Employs an upper-triangular additive mask filled with $-\infty$ (`torch.triu(..., diagonal=1)`) to enforce autoregressive attention constraints.
- **Custom Activations & Normalization:** Implements `QuickGELU` ($x \cdot \sigma(1.702 \cdot x)$) and an FP16-safe `LayerNorm` subclass.
- **`[EOS]` Token Pooling:** Extracts representations from the index of the highest non-zero token (`text.argmax(dim=-1)`), projected into the joint embedding space via a learned matrix `text_projection`.

### 3. Contrastive Objective & Loss (`train.py`)
- **Cosine Normalization:** Enforces $L_2$ normalization on both visual vectors ($v$) and textual vectors ($u$) before similarity computation.
- **Learnable Temperature:** Maintains a log-parameterized scale factor $\text{logit\_scale}$, initialized to $\ln(1 / 0.07)$ and clamped at $\tau \le 100$ for numerical stability.
- **Symmetric Cross-Entropy (InfoNCE):** Computes bidirectional cross-entropy along both rows (image-to-text) and columns (text-to-image) against diagonal identity targets.
- **Mixed-Precision Optimization:** Uses `torch.autocast` and `torch.cuda.amp.GradScaler` coupled with the `AdamW` optimizer and a `CosineAnnealingLR` schedule.

### 4. Dataset & Preprocessing (`coco_dataset.py`)
- Integrates with `torchvision.datasets.CocoCaptions`.
- Preprocesses images with Bicubic resizing to $224 \times 224$, center cropping, and standard CLIP image normalization:
  - $\mu = [0.48145466, 0.4578275, 0.40821073]$
  - $\sigma = [0.26862954, 0.26130258, 0.27577711]$
- Randomly samples 1 caption per image during each epoch, tokenizing it with Hugging Face's `CLIPTokenizer` (`openai/clip-vit-base-patch32`) padded/truncated to 77 tokens.

---

## Mathematical Formulation

### Patch Partitioning & Linear Projection
Given an image $I \in \mathbb{R}^{3 \times 224 \times 224}$ and patch resolution $P = 16$, the image is decomposed into $N = \frac{224 \times 224}{16 \times 16} = 196$ patches:

$$z_0 = \left[ x_{\text{cls}} \,;\, x_p^1 E \,;\, x_p^2 E \,;\, \dots \,;\, x_p^N E \right] + E_{\text{pos}}$$

where $E \in \mathbb{R}^{(P^2 \cdot 3) \times D}$ and $E_{\text{pos}} \in \mathbb{R}^{(N + 1) \times D}$.

### QuickGELU Activation Function
The text encoder utilizes the fast sigmoid approximation of the Gaussian Error Linear Unit:

$$\text{QuickGELU}(x) = x \odot \sigma(1.702 \cdot x)$$

### Autoregressive Attention Masking
To prevent positions from attending to subsequent tokens:

$$M_{i, j} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

### Symmetric Cross-Entropy Loss
For a batch of $B$ normalized visual features $v_i \in \mathbb{R}^{D}$ and textual features $u_j \in \mathbb{R}^{D}$:

$$S_{i, j} = \tau \cdot (v_i \cdot u_j^T), \quad \text{where } \tau = \min\left(e^{\text{logit\_scale}}, 100\right)$$

$$\mathcal{L}_{\text{image}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(S_{i, i})}{\sum_{j=1}^{B} \exp(S_{i, j})}, \quad \mathcal{L}_{\text{text}} = -\frac{1}{B} \sum_{j=1}^{B} \log \frac{\exp(S_{j, j})}{\sum_{i=1}^{B} \exp(S_{i, j})}$$

$$\mathcal{L}_{\text{total}} = \frac{\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}}{2}$$

---

## Installation

### Dependencies
Install the required dependencies using `pip`:

```bash
pip install torch torchvision einops transformers pandas pillow
```

Ensure your CUDA runtime and PyTorch match if training on GPU:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Training Pipeline

### Dataset Preparation
Download the **MS COCO Captions 2017** dataset:
1. Images: `train2017.zip`
2. Annotations: `annotations_trainval2017.zip`

Extract them to a directory structured as follows:
```
coco/
├── images/
│   ├── 000000000009.jpg
│   └── ...
└── annotations/
    └── captions_train2017.json
```

### Running the Training Script
The script `train.py` accepts positional arguments configured via `argparse`:

```bash
python train.py `<epochs>` `<batch_size>` `<image_folder>`
```

#### Example Command:
```bash
python train.py \
    10 \
    128 \
    5e-4 \
    0.01 \
    512 \
    50 \
    /path/to/coco/images \
    /path/to/coco/annotations/captions_train2017.json
```

#### Default Architectural Settings Initialized in `train.py`:
* **Image Encoder (`ViT`):** `image_size=224`, `patch_size=16`, `dim=embed_dim`, `layer=12`, `heads=8`, `mlp_dim=1024`
* **Text Encoder (`TextEncoder`):** `context_length=77`, `vocab_size=49408`, `embed_dim=embed_dim`, `transformer_width=768`, `transformer_heads=8`, `transformer_layers=12`
* **Tokenizer:** `openai/clip-vit-base-patch32` via Hugging Face `CLIPTokenizer`
* **Checkpoints:** Automatically saved at the end of every epoch as `clip_epoch_{epoch+1}.pt` containing the model state, optimizer state, epoch index, and average loss.

---

## Testing & Verification

You can verify the standalone `text_encoder.py` module directly:

```bash
python text_encoder.py
```

This runs the built-in test block to verify that the causal attention mask, positional embeddings, and `[EOS]` token pooling output the expected tensor dimension `[batch_size, embed_dim]`:
```text
Text Encoder architecture created successfully.
Total parameters: 63,443,456

--- Input ---
Dummy input shape: torch.Size([4, 76])
...
--- Output ---
Output embedding shape: torch.Size([4, 512])
This shape is [batch_size, embed_dim], as expected.
```

---

## License

This project is licensed under the MIT License.
