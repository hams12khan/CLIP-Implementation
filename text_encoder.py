import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    This is a common practice for training with mixed precision.
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    """
    A faster GELU activation function approximation.
    Used in the original CLIP implementation.
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    """
    A single block of the Transformer, containing a Multi-Head Attention
    layer and a Feed-Forward MLP, with residual connections.
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # Multi-Head Self-Attention layer
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # Layer normalization for the attention layer output
        self.ln_1 = LayerNorm(d_model)
        
        # Feed-Forward MLP (Multi-Layer Perceptron)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        # Layer normalization for the MLP output
        self.ln_2 = LayerNorm(d_model)
        
        # The attention mask to be used for causal (masked) self-attention
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # The attention mask is passed to the multi-head attention layer.
        # This ensures the model can't "look ahead" at future tokens,
        # as specified in the paper.
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Residual connection around the attention block
        x = x + self.attention(self.ln_1(x))
        # Residual connection around the MLP block
        x = x + self.mlp(self.ln_2(x))
        return x

class TextEncoder(nn.Module):
    """
    The main Text Encoder class for CLIP. It combines the token embeddings,
    positional embeddings, and multiple Transformer blocks.
    """
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 embed_dim: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()

        self.context_length = context_length
        self.transformer_width = transformer_width
        
        # --- Embedding Layers ---
        # Token embedding layer: maps token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # Positional embedding layer: learns a vector for each position
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        nn.init.normal_(self.positional_embedding, std=0.01)

        # --- Transformer ---
        # The core of the model: a stack of ResidualAttentionBlocks
        self.transformer = nn.Sequential(*[
            ResidualAttentionBlock(transformer_width, transformer_heads, attn_mask=self.build_attention_mask())
            for _ in range(transformer_layers)
        ])

        # --- Final Layers ---
        # Layer normalization for the final output
        self.ln_final = LayerNorm(transformer_width)
        
        # The final linear projection layer that maps the text features
        # into the multi-modal embedding space.
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)

    def build_attention_mask(self):
        """
        Builds the causal attention mask. This prevents the model from attending
        to future tokens, which is a key feature of autoregressive models like GPT
        and was used in the CLIP text encoder.
        """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # Zero out the lower triangle, leaving -inf in the upper triangle
        return mask

    def forward(self, text: torch.Tensor):
        # text shape: [batch_size, context_length]
        
        # 1. Get token and positional embeddings
        x = self.token_embedding(text)  # [batch_size, context_length, width]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND (required by PyTorch's Transformer)
        
        # 2. Pass through the Transformer blocks
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 3. Final layer normalization
        x = self.ln_final(x)

        # 4. Extract the features for the [EOS] token
        # The paper specifies that the features of the highest layer of the
        # transformer at the [EOS] token are used as the final representation.
        # We find the index of the [EOS] token for each sequence in the batch.
        # `torch.argmax` is used to find the last non-zero token index,
        # which corresponds to the [EOS] token.
        eos_token_indices = text.argmax(dim=-1)
        
        # Gather the features at the [EOS] token positions
        # x[torch.arange(x.shape[0]), eos_token_indices] -> shape [batch_size, transformer_width]
        x = x[torch.arange(x.shape[0]), eos_token_indices]
        
        # 5. Apply the final linear projection
        x = x @ self.text_projection

        return x

if __name__ == '__main__':
    # --- Model Configuration (matches the paper's base model) ---
    CONTEXT_LENGTH = 76  # Max sequence length
    VOCAB_SIZE = 49152   # From the BPE tokenizer
    EMBED_DIM = 512      # Dimension of the final multi-modal embedding space
    TRANSFORMER_WIDTH = 768  # Width of the transformer layers
    TRANSFORMER_HEADS = 8
    TRANSFORMER_LAYERS = 12

    # --- Instantiate the Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder = TextEncoder(
        context_length=CONTEXT_LENGTH,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        transformer_width=TRANSFORMER_WIDTH,
        transformer_heads=TRANSFORMER_HEADS,
        transformer_layers=TRANSFORMER_LAYERS
    ).to(device)

    print("Text Encoder architecture created successfully.")
    print(f"Total parameters: {sum(p.numel() for p in text_encoder.parameters()):,}") # Should be ~63M

    # --- Example Usage ---
    # Create a dummy batch of tokenized text
    # Let's assume 0 is the padding token and 49151 is the [EOS] token.
    # Each sequence has a different length.
    dummy_text = torch.randint(1, 49150, (4, CONTEXT_LENGTH), dtype=torch.long, device=device)
    dummy_text[:, 10] = 49151 # Add EOS token at index 10 for first sequence
    dummy_text[1, 20:] = 0    # Pad the second sequence
    dummy_text[1, 19] = 49151 # Add EOS token
    dummy_text[2, 40:] = 0    # Pad the third sequence
    dummy_text[2, 39] = 49151 # Add EOS token
    dummy_text[3, 60:] = 0    # Pad the fourth sequence
    dummy_text[3, 59] = 49151 # Add EOS token

    print("\n--- Input ---")
    print(f"Dummy input shape: {dummy_text.shape}")
    print("Example input sequence (first item in batch):")
    print(dummy_text[0])

    # --- Forward Pass ---
    with torch.no_grad():
        output_embeddings = text_encoder(dummy_text)

    print("\n--- Output ---")
    print(f"Output embedding shape: {output_embeddings.shape}")
    print("This shape is [batch_size, embed_dim], as expected.")

