import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig
from typing import Optional, Tuple, Dict, Any
import math


class DistilBERTTextEncoder(nn.Module):
    """
    Text encoder using DistilBERT for Tiny-MultiModal-Larimar.
    Handles text input and provides textual features for multimodal processing.
    """

    def __init__(self,
                 model_name: str = "distilbert-base-uncased",
                 latent_size: int = 384,
                 freeze_backbone: bool = False,
                 add_projection: bool = True,
                 max_length: int = 512):
        super(DistilBERTTextEncoder, self).__init__()

        self.latent_size = latent_size
        self.freeze_backbone = freeze_backbone
        self.max_length = max_length

        # Load DistilBERT model
        self.text_model = DistilBertModel.from_pretrained(model_name)
        self.text_config = self.text_model.config

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.text_model.parameters():
                param.requires_grad = False

        # DistilBERT outputs 768-dimensional features
        self.hidden_size = self.text_config.hidden_size  # 768

        # Optional projection layer to match latent size
        if add_projection and self.hidden_size != latent_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, latent_size),
                nn.LayerNorm(latent_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        else:
            self.projection = None

        # Linear layer for VAE-style latent encoding (mean and logvar)
        self.latent_encoder = nn.Linear(latent_size, latent_size * 2)

        # Layer norm for output
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False,
                return_sequence_features: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through text encoder
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            output_hidden_states: Whether to return all hidden states
            return_sequence_features: Whether to return sequence-level features
        Returns:
            pooled_output: [batch_size, latent_size] - Global text representation
            sequence_features: [batch_size, seq_len, latent_size] - Token-level features (optional)
        """
        batch_size = input_ids.size(0)

        # Pass through DistilBERT
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        # Get last hidden state: [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs.last_hidden_state

        # Apply projection if needed
        if self.projection is not None:
            last_hidden_state = self.projection(last_hidden_state)

        # Extract global representation using attention-weighted pooling
        if attention_mask is not None:
            # Mask out padded tokens
            mask_expanded = attention_mask.unsqueeze(
                -1).expand_as(last_hidden_state)
            masked_hidden_state = last_hidden_state * mask_expanded

            # Compute weighted average
            sum_embeddings = torch.sum(masked_hidden_state, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1, keepdim=True)
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple mean pooling
            pooled_output = last_hidden_state.mean(dim=1)

        # Apply layer norm
        pooled_output = self.layer_norm(pooled_output)

        if return_sequence_features:
            sequence_features = self.layer_norm(last_hidden_state)
            return pooled_output, sequence_features
        else:
            return pooled_output, None

    def encode_latent(self, input_ids: torch.Tensor,
                      attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text to latent space with mean and logvar for VAE
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
        """
        pooled_output, _ = self.forward(input_ids, attention_mask)

        # Compute mean and logvar
        mean_logvar = self.latent_encoder(pooled_output)
        mean, logvar = mean_logvar.chunk(2, dim=-1)

        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor,
                       num_samples: int = 1) -> torch.Tensor:
        """
        Reparameterization trick for VAE
        Args:
            mean: [batch_size, latent_size]
            logvar: [batch_size, latent_size]
            num_samples: Number of samples to draw
        Returns:
            z: [batch_size, num_samples, latent_size]
        """
        batch_size, latent_size = mean.size()
        std = torch.exp(0.5 * logvar)

        # Expand for multiple samples
        mean_expanded = mean.unsqueeze(1).expand(
            batch_size, num_samples, latent_size)
        std_expanded = std.unsqueeze(1).expand(
            batch_size, num_samples, latent_size)

        # Sample from standard normal
        eps = torch.randn_like(std_expanded)

        # Reparameterize
        z = mean_expanded + eps * std_expanded

        return z

    def freeze_encoder(self):
        """Freeze the text encoder backbone"""
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.freeze_backbone = True

    def unfreeze_encoder(self):
        """Unfreeze the text encoder backbone"""
        for param in self.text_model.parameters():
            param.requires_grad = True
        self.freeze_backbone = False


class TextProcessor(nn.Module):
    """
    Text processing utilities for handling various text inputs
    """

    def __init__(self, max_length: int = 512):
        super(TextProcessor, self).__init__()
        self.max_length = max_length

    def create_attention_mask(self, input_ids: torch.Tensor,
                              pad_token_id: int = 0) -> torch.Tensor:
        """
        Create attention mask for input_ids
        Args:
            input_ids: [batch_size, seq_len]
            pad_token_id: ID of padding token
        Returns:
            attention_mask: [batch_size, seq_len]
        """
        return (input_ids != pad_token_id).long()

    def truncate_sequences(self, input_ids: torch.Tensor,
                           attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Truncate sequences to max_length
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            truncated_input_ids: [batch_size, max_length]
            truncated_attention_mask: [batch_size, max_length]
        """
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_length]

        return input_ids, attention_mask


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence features
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x + pe: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class TextEmbeddingLayer(nn.Module):
    """
    Custom text embedding layer with positional encoding
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512,
                 dropout: float = 0.1, padding_idx: int = 0):
        super(TextEmbeddingLayer, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through embedding layer
        Args:
            input_ids: [batch_size, seq_len]
        Returns:
            embedded: [batch_size, seq_len, d_model]
        """
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        embedded = self.pos_encoding(embedded)
        return self.dropout(embedded)
