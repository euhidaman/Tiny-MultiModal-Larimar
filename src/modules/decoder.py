import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, AutoModelForCausalLM
from typing import Optional, Tuple, Dict, Any, List
import math


class UnifiedGPT2Decoder(nn.Module):
    """
    Unified decoder using GPT-2 variants for Tiny-MultiModal-Larimar.
    Supports DistilGPT-2, GPT2-medium, and other GPT-2 variants.
    Handles text generation conditioned on multimodal latent representations.
    """

    def __init__(self,
                 model_name: str = "gpt2-medium",
                 latent_size: int = 384,
                 vocab_size: int = 50257,
                 max_length: int = 512,
                 add_latent_conditioning: bool = True):
        super(UnifiedGPT2Decoder, self).__init__()

        self.model_name = model_name
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.add_latent_conditioning = add_latent_conditioning

        # Load GPT-2 model using AutoModel for flexibility
        self.gpt2_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.gpt2_config = self.gpt2_model.config

        # GPT-2 hidden size
        self.hidden_size = self.gpt2_config.hidden_size  # 768

        # Latent conditioning layers
        if add_latent_conditioning:
            # Project latent to GPT-2 hidden size
            self.latent_projection = nn.Linear(latent_size, self.hidden_size)

            # Conditioning layers for each transformer layer
            self.num_layers = self.gpt2_config.n_layer
            self.latent_conditioning = nn.ModuleList([
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ])

            # Layer norm for latent conditioning
            self.latent_norm = nn.LayerNorm(self.hidden_size)

        # Special tokens
        self.pad_token_id = self.gpt2_config.pad_token_id if self.gpt2_config.pad_token_id is not None else self.gpt2_config.eos_token_id
        self.bos_token_id = self.gpt2_config.bos_token_id if self.gpt2_config.bos_token_id is not None else self.gpt2_config.eos_token_id
        self.eos_token_id = self.gpt2_config.eos_token_id

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple] = None,
                labels: Optional[torch.Tensor] = None,
                latent_conditioning: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decoder
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            past_key_values: Cached key-value pairs
            labels: [batch_size, seq_len] - for computing loss
            latent_conditioning: [batch_size, latent_size] - multimodal conditioning
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return as dict
        Returns:
            outputs: Dictionary containing logits, loss, etc.
        """
        batch_size = input_ids.size(0)

        # Prepare latent conditioning if provided
        if latent_conditioning is not None and self.add_latent_conditioning:
            # Project latent to hidden size
            latent_projected = self.latent_projection(latent_conditioning)
            latent_projected = self.latent_norm(latent_projected)

            # Expand for sequence length
            seq_len = input_ids.size(1)
            latent_expanded = latent_projected.unsqueeze(
                1).expand(batch_size, seq_len, self.hidden_size)
        else:
            latent_expanded = None

        # If we have latent conditioning, we need to modify the forward pass
        if latent_expanded is not None:
            # Get embeddings from GPT-2
            inputs_embeds = self.gpt2_model.transformer.wte(input_ids)

            # Add latent conditioning to input embeddings
            inputs_embeds = inputs_embeds + latent_expanded

            # Forward pass through GPT-2 with modified embeddings
            outputs = self.gpt2_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            # Standard forward pass
            outputs = self.gpt2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return outputs

    def generate(self, input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 latent_conditioning: Optional[torch.Tensor] = None,
                 max_length: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 do_sample: bool = True,
                 num_return_sequences: int = 1,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate text with latent conditioning
        Args:
            input_ids: [batch_size, seq_len] - input prompt
            attention_mask: [batch_size, seq_len]
            latent_conditioning: [batch_size, latent_size] - multimodal conditioning
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        Returns:
            generated_ids: [batch_size * num_return_sequences, generated_length]
        """
        if max_length is None:
            max_length = self.max_length

        if pad_token_id is None:
            pad_token_id = self.pad_token_id

        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        # Prepare generation parameters
        generation_config = {
            'max_length': max_length,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'do_sample': do_sample,
            'num_return_sequences': num_return_sequences,
            'pad_token_id': pad_token_id,
            'eos_token_id': eos_token_id,
            'use_cache': True
        }

        # If we have latent conditioning, we need custom generation
        if latent_conditioning is not None and self.add_latent_conditioning:
            return self._generate_with_latent_conditioning(
                input_ids=input_ids,
                attention_mask=attention_mask,
                latent_conditioning=latent_conditioning,
                **generation_config
            )
        else:
            # Standard generation
            return self.gpt2_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

    def _generate_with_latent_conditioning(self, input_ids: torch.Tensor,
                                           attention_mask: Optional[torch.Tensor],
                                           latent_conditioning: torch.Tensor,
                                           **generation_config) -> torch.Tensor:
        """
        Custom generation with latent conditioning
        """
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        max_length = generation_config['max_length']
        temperature = generation_config['temperature']
        top_k = generation_config['top_k']
        top_p = generation_config['top_p']
        do_sample = generation_config['do_sample']
        pad_token_id = generation_config['pad_token_id']
        eos_token_id = generation_config['eos_token_id']

        # Initialize output
        generated_ids = input_ids.clone()

        # Generate tokens one by one
        for i in range(seq_len, max_length):
            # Get current attention mask
            current_attention_mask = attention_mask
            if current_attention_mask is not None:
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones(batch_size, 1,
                               device=current_attention_mask.device)
                ], dim=1)

            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=current_attention_mask,
                latent_conditioning=latent_conditioning,
                use_cache=False
            )

            # Get next token logits
            next_token_logits = outputs.logits[:, -1, :] / temperature

            # Apply top-k and top-p filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, top_k)
                next_token_logits = torch.full_like(
                    next_token_logits, -float('inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample next token
            if do_sample:
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(
                    next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            # Update attention mask
            if attention_mask is not None:
                attention_mask = current_attention_mask

            # Check for EOS token
            if torch.all(next_token == eos_token_id):
                break

        return generated_ids

    def compute_loss(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None,
                     latent_conditioning: Optional[torch.Tensor] = None,
                     reduction: str = 'mean') -> torch.Tensor:
        """
        Compute language modeling loss
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
            latent_conditioning: [batch_size, latent_size]
            reduction: Loss reduction method
        Returns:
            loss: Scalar loss
        """
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            latent_conditioning=latent_conditioning
        )

        loss = outputs.loss

        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings"""
        self.gpt2_model.resize_token_embeddings(new_num_tokens)
        self.vocab_size = new_num_tokens

    def get_output_embeddings(self):
        """Get output embeddings"""
        return self.gpt2_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        """Set output embeddings"""
        self.gpt2_model.set_output_embeddings(new_embeddings)


class LatentToText(nn.Module):
    """
    Module to convert latent representations to text conditioning
    """

    def __init__(self, latent_size: int, hidden_size: int, num_layers: int = 2):
        super(LatentToText, self).__init__()

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(latent_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))

            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))

        self.projection = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Convert latent to text conditioning
        Args:
            latent: [batch_size, latent_size]
        Returns:
            conditioning: [batch_size, hidden_size]
        """
        projected = self.projection(latent)
        return self.layer_norm(projected)


# Backward compatibility alias
DistilGPT2Decoder = UnifiedGPT2Decoder
