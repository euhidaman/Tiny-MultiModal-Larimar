import torch
import torch.nn as nn
import numpy as np
import math
from typing import Tuple, Optional

EPSILON = 1e-6


class TinyMemory(nn.Module):
    """
    Simplified version of GPM (Generative Parametric Memory) for Tiny-MultiModal-Larimar.
    Reduces memory size and computational complexity while maintaining core episodic memory functionality.
    """

    def __init__(self,
                 code_size: int = 384,
                 memory_size: int = 128,
                 direct_writing: bool = True,
                 ordering: bool = False,
                 pseudoinverse_approx_step: int = 8,
                 observation_noise_std: float = 0.000001,
                 identity: bool = False,
                 w_logvar_setting: int = 0,
                 deterministic: bool = False,
                 device: int = 0):

        super(TinyMemory, self).__init__()

        self._code_size = code_size
        self._memory_size = memory_size
        self._direct_writing = direct_writing
        self._ordering = ordering
        self._pseudoinverse_approx_step = pseudoinverse_approx_step
        self._observation_noise_std = observation_noise_std
        self._w_logvar_setting = w_logvar_setting
        self.deterministic = deterministic

        # Memory parameters
        if identity:
            self.memory_logvar = nn.Parameter(
                torch.zeros(1), requires_grad=False)
            self.memory_mean = nn.Parameter(torch.eye(min(self._memory_size, self._code_size),
                                                      self._code_size), requires_grad=True)
            if self._memory_size > self._code_size:
                # Pad with zeros if memory_size > code_size
                padding = torch.zeros(
                    self._memory_size - self._code_size, self._code_size)
                self.memory_mean = nn.Parameter(
                    torch.cat([self.memory_mean, padding], dim=0), requires_grad=True)
        else:
            self.memory_logvar = nn.Parameter(
                torch.zeros(1), requires_grad=False)
            self.memory_mean = nn.Parameter(torch.randn(
                self._memory_size, self._code_size), requires_grad=True)

        # W logvar parameters (simplified)
        if w_logvar_setting == 0:
            self.w_logvar = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif w_logvar_setting == 1:
            self.w_logvar = nn.Parameter(torch.zeros(
                self._memory_size), requires_grad=True)
        elif w_logvar_setting == 2:
            self.w_logvar = nn.Linear(self._code_size, self._memory_size)
        else:
            self.w_logvar = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Ordering (simplified LSTM)
        if self._ordering:
            self.lstm_z = nn.LSTM(input_size=self._code_size,
                                  hidden_size=self._code_size//2,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)

        # Initialize memory
        self.register_buffer('ben_cohen_memory_init', torch.tensor([-5.0]))

    def get_prior_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prior parameters for memory distribution"""
        prior_var = torch.ones(
            self._memory_size, device=self.memory_mean.device) * torch.exp(self.memory_logvar) + EPSILON
        prior_cov = torch.diag(prior_var)
        return self.memory_mean, prior_cov

    def write_to_memory(self, input_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write encoded input to memory
        Args:
            input_encoded: [batch_size, seq_len, code_size] or [batch_size, code_size]
        Returns:
            posterior_memory: Updated memory state
            dkl_M: KL divergence for memory update
        """
        batch_size = input_encoded.size(0)

        # Handle different input shapes
        if input_encoded.dim() == 3:
            # Average over sequence dimension
            input_encoded = input_encoded.mean(
                dim=1)  # [batch_size, code_size]

        # Get prior parameters
        memory_mean, memory_cov = self.get_prior_params()

        # Simplified memory update using direct writing
        if self._direct_writing:
            # Direct writing: update memory with current input
            posterior_memory = memory_mean.unsqueeze(0).expand(
                batch_size, -1, -1)  # [batch_size, memory_size, code_size]

            # Simple update rule: weighted average
            alpha = 0.1  # Learning rate for memory update
            memory_update = input_encoded.unsqueeze(
                1)  # [batch_size, 1, code_size]

            # Find closest memory slot and update it
            # [batch_size, 1, memory_size]
            similarities = torch.bmm(
                memory_update, posterior_memory.transpose(1, 2))
            closest_idx = similarities.squeeze(1).argmax(dim=1)  # [batch_size]

            # Update the closest memory slot
            batch_indices = torch.arange(
                batch_size, device=input_encoded.device)
            posterior_memory[batch_indices, closest_idx] = (
                1 - alpha) * posterior_memory[batch_indices, closest_idx] + alpha * input_encoded

            # Compute KL divergence (simplified)
            dkl_M = torch.sum(
                (input_encoded - memory_mean[closest_idx])**2, dim=1) * 0.5
        else:
            # Indirect writing using pseudoinverse approximation
            posterior_memory = memory_mean.unsqueeze(
                0).expand(batch_size, -1, -1)
            dkl_M = torch.zeros(batch_size, device=input_encoded.device)

        return posterior_memory, dkl_M

    def read_with_encoded_input(self,
                                input_encoded: torch.Tensor,
                                memory_state: torch.Tensor,
                                reduce_kl: bool = True,
                                get_w: bool = False,
                                deterministic: bool = False,
                                sigma_w: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory given encoded input
        Args:
            input_encoded: [batch_size, code_size]
            memory_state: [batch_size, memory_size, code_size]
        Returns:
            z_read: Retrieved latent code
            kl_div: KL divergence for reading
        """
        batch_size = input_encoded.size(0)

        # Compute attention weights over memory
        query = input_encoded.unsqueeze(1)  # [batch_size, 1, code_size]
        # [batch_size, 1, memory_size]
        attention_scores = torch.bmm(query, memory_state.transpose(1, 2))
        attention_weights = torch.softmax(
            attention_scores / math.sqrt(self._code_size), dim=2)  # [batch_size, 1, memory_size]

        # Read from memory using attention weights
        z_read = torch.bmm(attention_weights, memory_state).squeeze(
            1)  # [batch_size, code_size]

        # Compute KL divergence for reading
        if reduce_kl:
            kl_div = torch.sum((z_read - input_encoded)**2, dim=1) * 0.5
        else:
            kl_div = (z_read - input_encoded)**2 * 0.5

        if get_w:
            return z_read, kl_div, attention_weights.squeeze(1)
        else:
            return z_read, kl_div

    def forward(self, input_encoded: torch.Tensor,
                memory_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through memory
        Args:
            input_encoded: [batch_size, code_size]
            memory_state: Optional existing memory state
        Returns:
            z_read: Retrieved latent code
            updated_memory: Updated memory state
            kl_div: Total KL divergence
        """
        if memory_state is None:
            # Initialize memory state
            batch_size = input_encoded.size(0)
            memory_mean, _ = self.get_prior_params()
            memory_state = memory_mean.unsqueeze(0).expand(batch_size, -1, -1)

        # Write to memory
        updated_memory, dkl_write = self.write_to_memory(input_encoded)

        # Read from memory
        z_read, dkl_read = self.read_with_encoded_input(
            input_encoded, updated_memory)

        # Total KL divergence
        kl_div = dkl_write + dkl_read

        return z_read, updated_memory, kl_div
