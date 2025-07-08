import torch
import torch.nn as nn
from transformers import (
    GPT2LMHeadModel, GPT2Config, GPT2Model,
    AutoModelForCausalLM, AutoConfig
)
from typing import Optional, Tuple, Dict, Any, List, Union
import warnings
import math


class GPT2ForLatentConnector(nn.Module):
    """
    GPT-2 decoder for latent space connection, based on original Larimar architecture.
    This replaces the DistilGPT2 decoder with the original Larimar GPT2-based decoder.
    """

    def __init__(self,
                 model_name: str = "gpt2-medium",
                 latent_size: int = 384,
                 latent_as_gpt_emb: bool = True,
                 latent_as_gpt_memory: bool = True):
        super(GPT2ForLatentConnector, self).__init__()

        self.model_name = model_name
        self.latent_size = latent_size
        self.latent_as_gpt_emb = latent_as_gpt_emb
        self.latent_as_gpt_memory = latent_as_gpt_memory

        # Load GPT-2 model and config
        self.config = GPT2Config.from_pretrained(model_name)
        self.transformer = GPT2Model.from_pretrained(
            model_name, config=self.config)
        self.lm_head = nn.Linear(
            self.config.n_embd, self.config.vocab_size, bias=False)

        # Model parallel support
        self.model_parallel = False
        self.device_map = None

        # Latent conditioning components
        if latent_as_gpt_emb or latent_as_gpt_memory:
            # Project latent to GPT-2 embedding dimension
            self.latent_to_emb = nn.Linear(latent_size, self.config.n_embd)

            # Layer norm for latent conditioning
            self.latent_norm = nn.LayerNorm(self.config.n_embd)

        print(f"Initialized GPT2ForLatentConnector with {model_name}")
        print(
            f"Embedding dim: {self.config.n_embd}, Latent size: {latent_size}")
        print(
            f"Latent as embedding: {latent_as_gpt_emb}, Latent as memory: {latent_as_gpt_memory}")

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        """Prepare inputs for generation"""
        token_type_ids = kwargs.get("token_type_ids", None)

        # Only last token for input_ids if past is defined
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # Create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # If inputs_embeds are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                latent_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GPT-2 decoder with latent conditioning
        Args:
            input_ids: [batch_size, seq_len]
            past_key_values: Cached key-value pairs
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for computing loss
            latent_input: [batch_size, latent_size] - latent conditioning
            ... (other standard GPT-2 arguments)
        Returns:
            Dictionary containing logits, loss, etc.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Handle latent conditioning
        latent_embeds = None
        if latent_input is not None and (self.latent_as_gpt_emb or self.latent_as_gpt_memory):
            # Project latent to embedding dimension
            latent_embeds = self.latent_to_emb(latent_input)
            latent_embeds = self.latent_norm(latent_embeds)

            # Expand latent to match sequence length if using as memory
            if self.latent_as_gpt_memory and input_ids is not None:
                seq_len = input_ids.size(1)
                latent_embeds = latent_embeds.unsqueeze(
                    1).expand(-1, seq_len, -1)

        # Modify inputs_embeds if using latent as embedding
        if latent_embeds is not None and self.latent_as_gpt_emb and inputs_embeds is None:
            # Get token embeddings
            if input_ids is not None:
                inputs_embeds = self.transformer.wte(input_ids)
                # Add latent conditioning to embeddings
                if latent_embeds.dim() == 2:
                    latent_embeds = latent_embeds.unsqueeze(
                        1).expand(-1, inputs_embeds.size(1), -1)
                inputs_embeds = inputs_embeds + latent_embeds

        # Pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids if inputs_embeds is None else None,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # Apply language modeling head
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Move labels to correct device
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': lm_logits,
            'past_key_values': transformer_outputs.past_key_values,
            'hidden_states': transformer_outputs.hidden_states,
            'attentions': transformer_outputs.attentions,
        }

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """Reorder cache for beam search"""
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device))
                  for past_state in layer_past)
            for layer_past in past_key_values
        )


class LarimarGPT2Decoder(nn.Module):
    """
    Wrapper for Larimar-style GPT2 decoding compatible with the multimodal architecture.
    This maintains compatibility with the existing codebase while using Larimar components.
    """

    def __init__(self,
                 model_name: str = "gpt2-medium",
                 latent_size: int = 384,
                 vocab_size: int = 50257,
                 max_length: int = 512,
                 latent_as_gpt_emb: bool = True,
                 latent_as_gpt_memory: bool = True):
        super(LarimarGPT2Decoder, self).__init__()

        self.model_name = model_name
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Use GPT2ForLatentConnector as the core decoder
        self.decoder = GPT2ForLatentConnector(
            model_name=model_name,
            latent_size=latent_size,
            latent_as_gpt_emb=latent_as_gpt_emb,
            latent_as_gpt_memory=latent_as_gpt_memory
        )

        # Get config for compatibility
        self.config = self.decoder.config
        self.hidden_size = self.config.n_embd

        # Special tokens
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id
        self.bos_token_id = self.config.bos_token_id if self.config.bos_token_id is not None else self.config.eos_token_id
        self.eos_token_id = self.config.eos_token_id

        # Additional projection for multimodal conditioning
        self.multimodal_latent_projection = nn.Linear(latent_size, latent_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[Tuple] = None,
                labels: Optional[torch.Tensor] = None,
                latent_conditioning: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                return_dict: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with multimodal architecture
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            past_key_values: Cached key-value pairs
            labels: [batch_size, seq_len] - for computing loss
            latent_conditioning: [batch_size, latent_size] - multimodal conditioning
            ... (other arguments)
        Returns:
            outputs: Dictionary containing logits, loss, etc.
        """
        # Process latent conditioning
        processed_latent = None
        if latent_conditioning is not None:
            processed_latent = self.multimodal_latent_projection(
                latent_conditioning)
            processed_latent = self.dropout(processed_latent)

        # Pass through GPT2ForLatentConnector
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            latent_input=processed_latent,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return outputs

    def generate(self,
                 input_ids: torch.Tensor,
                 latent_conditioning: Optional[torch.Tensor] = None,
                 max_length: int = 50,
                 num_beams: int = 1,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 do_sample: bool = False,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 **kwargs) -> torch.Tensor:
        """
        Generate text conditioned on latent input
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.eos_token_id

        # Process latent conditioning
        processed_latent = None
        if latent_conditioning is not None:
            processed_latent = self.multimodal_latent_projection(
                latent_conditioning)
            processed_latent = self.dropout(processed_latent)

        # Simple generation loop (can be replaced with more sophisticated methods)
        batch_size = input_ids.size(0)
        generated = input_ids.clone()
        past_key_values = None

        for _ in range(max_length):
            outputs = self.decoder(
                input_ids=generated if past_key_values is None else generated[:, -1:],
                latent_input=processed_latent,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            logits = outputs['logits'][:, -1, :]  # Get last token logits
            past_key_values = outputs['past_key_values']

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, top_k_indices, top_k_logits)

            # Apply top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[...,
                                         1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample or take argmax
            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == eos_token_id).all():
                break

        return generated

    def resize_token_embeddings(self, new_num_tokens):
        """Resize token embeddings"""
        old_embeddings = self.decoder.transformer.wte
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.decoder.transformer.wte = new_embeddings

        # Also resize lm_head
        old_lm_head = self.decoder.lm_head
        new_lm_head = self._get_resized_lm_head(old_lm_head, new_num_tokens)
        self.decoder.lm_head = new_lm_head

        return new_embeddings

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens):
        """Helper to resize embeddings"""
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device,
                          dtype=old_embeddings.weight.dtype)

        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy,
                                   :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _get_resized_lm_head(self, old_lm_head, new_num_tokens):
        """Helper to resize lm_head"""
        old_num_tokens, old_embedding_dim = old_lm_head.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_lm_head

        # Create new lm_head
        new_lm_head = nn.Linear(
            old_embedding_dim, new_num_tokens, bias=old_lm_head.bias is not None)
        new_lm_head.to(old_lm_head.weight.device,
                       dtype=old_lm_head.weight.dtype)

        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_lm_head.weight.data[:num_tokens_to_copy,
                                :] = old_lm_head.weight.data[:num_tokens_to_copy, :]

        if new_lm_head.bias is not None:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]

        return new_lm_head
