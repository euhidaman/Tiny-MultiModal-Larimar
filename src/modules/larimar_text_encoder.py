import torch
import torch.nn as nn
from transformers import (
    BertModel, BertConfig, BertTokenizer,
    AutoModel, AutoConfig
)
from typing import Optional, Tuple, Dict, Any, List
import math


class BertForLatentConnector(nn.Module):
    """
    BERT encoder for latent space connection using Larimar architecture.
    Replaces DeBERTa encoder with original Larimar BERT-based approach.
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 latent_size: int = 384,
                 freeze_backbone: bool = False,
                 add_pooling_layer: bool = True):
        super(BertForLatentConnector, self).__init__()

        self.model_name = model_name
        self.latent_size = latent_size
        self.freeze_backbone = freeze_backbone

        # Load BERT model and config
        self.config = BertConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name, config=self.config)

        # Add pooler if specified
        self.pooler = self.bert.pooler if add_pooling_layer else None

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Linear layer for latent space projection (outputs mean and logvar)
        # This is the key component from original Larimar
        self.linear = nn.Linear(self.config.hidden_size,
                                2 * latent_size, bias=False)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)

        print(f"Initialized BertForLatentConnector with {model_name}")
        print(
            f"Hidden size: {self.config.hidden_size}, Latent size: {latent_size}")

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prune heads of the model."""
        for layer, heads in heads_to_prune.items():
            self.bert.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through BERT encoder
        Returns:
            pooled_output: [batch_size, hidden_size] - Pooled representation
            latent_params: [batch_size, 2 * latent_size] - Mean and logvar for VAE
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass through BERT
        encoder_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = encoder_outputs[1] if self.pooler is not None else None

        # If no pooled output, use [CLS] token
        if pooled_output is None:
            pooled_output = sequence_output[:, 0]  # [CLS] token

        # Apply layer norm
        pooled_output = self.layer_norm(pooled_output)

        # Project to latent space (mean and logvar)
        latent_params = self.linear(pooled_output)

        return pooled_output, latent_params

    def encode_to_latent(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Encode input to latent space with reparameterization
        Returns:
            latent_z: [batch_size, latent_size] - Sampled latent code
            mu: [batch_size, latent_size] - Mean
            logvar: [batch_size, latent_size] - Log variance
        """
        pooled_output, latent_params = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Split into mean and logvar
        mu, logvar = latent_params.chunk(2, -1)

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent_z = mu + eps * std

        return latent_z, mu, logvar


class LarimarTextEncoder(nn.Module):
    """
    Wrapper for Larimar-style text encoding compatible with the multimodal architecture.
    This maintains compatibility with the existing codebase while using Larimar components.
    """

    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 latent_size: int = 384,
                 freeze_backbone: bool = False):
        super(LarimarTextEncoder, self).__init__()

        self.model_name = model_name
        self.latent_size = latent_size

        # Use BertForLatentConnector as the core encoder
        self.encoder = BertForLatentConnector(
            model_name=model_name,
            latent_size=latent_size,
            freeze_backbone=freeze_backbone
        )

        # Additional layers for multimodal fusion compatibility
        self.multimodal_projection = nn.Linear(latent_size, latent_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                return_latent_params: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass compatible with multimodal architecture
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            return_latent_params: Whether to return VAE parameters
        Returns:
            text_features: [batch_size, latent_size] - Text representation
            latent_params: [batch_size, 2 * latent_size] - VAE parameters (optional)
        """
        # Get encodings from BERT
        pooled_output, latent_params = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # Extract mean from latent parameters for text features
        mu, logvar = latent_params.chunk(2, -1)

        # Use mean as the text representation (deterministic for inference)
        text_features = self.multimodal_projection(mu)
        text_features = self.dropout(text_features)

        if return_latent_params:
            return text_features, latent_params
        else:
            return text_features, None

    def encode_to_latent(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Encode to latent space with sampling
        """
        return self.encoder.encode_to_latent(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    def get_text_features(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Get deterministic text features (using mean)
        """
        text_features, _ = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_latent_params=False
        )
        return text_features


class RobertaForLatentConnector(nn.Module):
    """
    RoBERTa encoder for latent space connection, alternative to BERT.
    """

    def __init__(self,
                 model_name: str = "roberta-base",
                 latent_size: int = 384,
                 freeze_backbone: bool = False):
        super(RobertaForLatentConnector, self).__init__()

        self.model_name = model_name
        self.latent_size = latent_size
        self.freeze_backbone = freeze_backbone

        # Load RoBERTa model
        self.roberta = AutoModel.from_pretrained(model_name)
        self.config = self.roberta.config

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # Linear layer for latent space projection
        self.linear = nn.Linear(self.config.hidden_size,
                                2 * latent_size, bias=False)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)

        print(f"Initialized RobertaForLatentConnector with {model_name}")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through RoBERTa encoder"""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # Use [CLS] token (first token) as sentence representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.layer_norm(pooled_output)

        # Project to latent space
        latent_params = self.linear(pooled_output)

        return pooled_output, latent_params
