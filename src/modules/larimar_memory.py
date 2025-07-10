import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple, Optional

EPSILON = 1e-6


class TinyLarimarMemory(nn.Module):
    """
    Simplified memory module adapted from original Larimar for Tiny-MultiModal-Larimar.
    This provides episodic memory capabilities for multimodal learning.
    """

    def __init__(self,
                 code_size: int = 384,
                 memory_size: int = 512,
                 direct_writing: bool = True,
                 observation_noise_std: float = 0.1,
                 identity_init: bool = False,
                 w_logvar_setting: int = 0,
                 deterministic: bool = False):
        super(TinyLarimarMemory, self).__init__()

        self.code_size = code_size
        self.memory_size = memory_size
        self.observation_noise_std = observation_noise_std
        self.direct_writing = direct_writing
        self.deterministic = deterministic
        self.w_logvar_setting = w_logvar_setting

        # Memory parameters
        if identity_init:
            # Initialize as identity for better initial performance
            self.memory_logvar = nn.Parameter(
                torch.zeros(1), requires_grad=False)
            # Pad or truncate identity matrix to match dimensions
            if memory_size >= code_size:
                identity_mem = torch.eye(code_size)
                padding = torch.zeros(memory_size - code_size, code_size)
                self.memory_mean = nn.Parameter(
                    torch.cat([identity_mem, padding], dim=0), requires_grad=True)
            else:
                self.memory_mean = nn.Parameter(
                    torch.eye(memory_size, code_size), requires_grad=True)
        else:
            self.memory_logvar = nn.Parameter(
                torch.zeros(1), requires_grad=False)
            self.memory_mean = nn.Parameter(torch.randn(
                memory_size, code_size) * 0.1, requires_grad=True)

        # Attention weights variance
        if w_logvar_setting == 0:
            # Single variance for all dimensions
            self.w_logvar = nn.Parameter(torch.zeros(1), requires_grad=True)
        elif w_logvar_setting == 1:
            # Per-memory-slot variance
            self.w_logvar = nn.Parameter(
                torch.zeros(memory_size), requires_grad=True)
        elif w_logvar_setting == 2:
            # Learned from input
            self.w_logvar = nn.Linear(code_size, memory_size)
        else:
            # Default to single variance
            self.w_logvar = nn.Parameter(torch.zeros(1), requires_grad=True)

        # Pseudoinverse approximation parameters
        self.pseudoinverse_steps = 3
        self.ben_cohen_init = torch.tensor([-5.0])

        print(f"Initialized TinyLarimarMemory:")
        print(f"  Code size: {code_size}, Memory size: {memory_size}")
        print(
            f"  Direct writing: {direct_writing}, Identity init: {identity_init}")

    def get_prior_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prior memory parameters"""
        prior_var = torch.ones(
            self.memory_size, device=self.memory_mean.device) * torch.exp(self.memory_logvar) + EPSILON
        prior_cov = torch.diag(prior_var)
        return self.memory_mean, prior_cov

    def get_prior_state(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get prior memory state for batch"""
        prior_mean, prior_cov = self.get_prior_params()

        batch_prior_mean = prior_mean.unsqueeze(0).expand(batch_size, -1, -1)
        batch_prior_cov = prior_cov.unsqueeze(0).expand(batch_size, -1, -1)

        return batch_prior_mean, batch_prior_cov

    def sample_memory(self, memory_state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Sample memory from current state"""
        memory_mean, memory_cov = memory_state
        return memory_mean  # For simplicity, use mean (deterministic)

    def solve_attention_weights(self, z: torch.Tensor, M: torch.Tensor, pseudoinverse: bool = True) -> torch.Tensor:
        """
        Solve for attention weights w such that z ≈ w^T M
        Args:
            z: [episode_size, batch_size, code_size] or [batch_size, code_size]
            M: [batch_size, memory_size, code_size]
        Returns:
            w: [episode_size, batch_size, memory_size] or [batch_size, memory_size]
        """
        if z.dim() == 2:
            # Single timestep
            z = z.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        episode_size, batch_size, code_size = z.shape

        if pseudoinverse:
            # Use pseudoinverse for stable solution
            # [batch_size, code_size, memory_size]
            M_transposed = M.transpose(1, 2)

            # Solve w^T = z M^T (M M^T)^{-1}
            # [batch_size, memory_size, memory_size]
            MMT = torch.bmm(M, M_transposed)

            # Add small regularization for numerical stability
            reg = torch.eye(self.memory_size, device=MMT.device) * 1e-6
            MMT_reg = MMT + reg.unsqueeze(0)

            try:
                # [batch_size, memory_size, memory_size]
                MMT_inv = torch.inverse(MMT_reg)
            except:
                # Fallback to regularized version
                reg = torch.eye(self.memory_size, device=MMT.device) * 1e-3
                MMT_inv = torch.inverse(MMT + reg.unsqueeze(0))

            w_list = []
            for i in range(episode_size):
                z_step = z[i]  # [batch_size, code_size]
                # w = z M^T (M M^T)^{-1}
                w_step = torch.bmm(
                    torch.bmm(z_step.unsqueeze(1), M_transposed), MMT_inv)
                w_list.append(w_step.squeeze(1))  # [batch_size, memory_size]

            # [episode_size, batch_size, memory_size]
            w = torch.stack(w_list, dim=0)
        else:
            # Least squares solution
            w_list = []
            for i in range(episode_size):
                z_step = z[i].unsqueeze(-1)  # [batch_size, code_size, 1]
                # [batch_size, memory_size]
                w_step = torch.linalg.lstsq(M, z_step).solution.squeeze(-1)
                w_list.append(w_step)
            # [episode_size, batch_size, memory_size]
            w = torch.stack(w_list, dim=0)

        if squeeze_output:
            w = w.squeeze(0)

        return w

    def update_memory(self, old_memory: Tuple[torch.Tensor, torch.Tensor],
                      w: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update memory with new observations
        Args:
            old_memory: (mean, cov) of memory
            w: [1, batch_size, memory_size] attention weights
            z: [1, batch_size, code_size] observations
        Returns:
            new_memory: Updated (mean, cov)
        """
        old_mean, old_cov = old_memory

        # Prediction error
        pred_z = torch.bmm(w.transpose(0, 1), old_mean).transpose(
            0, 1)  # [1, batch_size, code_size]
        delta = z - pred_z

        # Update equations (simplified Kalman-like update)
        wU = torch.bmm(w.transpose(0, 1), old_cov).transpose(
            0, 1)  # [1, batch_size, memory_size]
        wUw = torch.bmm(wU.transpose(0, 1), w.transpose(
            0, 1).transpose(1, 2)).transpose(0, 1)  # [1, batch_size, 1]

        sigma_z = wUw + self.observation_noise_std**2 * \
            torch.eye(1, device=w.device).unsqueeze(0).expand_as(wUw)
        c_z = wU / (sigma_z + EPSILON)

        # Updated memory
        new_mean = old_mean + \
            torch.bmm(c_z.transpose(0, 1).transpose(
                1, 2), delta.transpose(0, 1))
        new_cov = old_cov - \
            torch.bmm(c_z.transpose(0, 1).transpose(1, 2), wU.transpose(0, 1))

        return new_mean, new_cov

    def write_to_memory(self, input_encoded: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Write encoded inputs to memory
        Args:
            input_encoded: [episode_size, batch_size, code_size]
        Returns:
            memory_state: Updated memory state
            dkl_memory: KL divergence of memory update
        """
        batch_size = input_encoded.shape[1]
        prior_memory = self.get_prior_state(batch_size)

        if self.direct_writing:
            # Direct writing approach (simpler and more stable)
            episode_size = input_encoded.shape[0]

            # Add noise for regularization
            noise = torch.randn_like(input_encoded) * \
                self.observation_noise_std
            z_noisy = input_encoded + noise

            # Solve for attention weights
            w = self.solve_attention_weights(
                z_noisy, prior_memory[0], pseudoinverse=True)

            # w is [episode_size, batch_size, memory_size] = [3, 2, 512]
            # We need to compute new_memory_mean as [batch_size, memory_size, code_size]
            
            # Simple approach: use least squares for each batch separately
            new_memory_means = []
            for b in range(batch_size):
                # Extract tensors for this batch: w[:, b, :] = [episode_size, memory_size] = [3, 512]
                w_batch = w[:, b, :]  # [episode_size, memory_size]
                z_batch = z_noisy[:, b, :]  # [episode_size, code_size]
                
                # Solve: w_batch^T @ memory_mean = z_batch^T
                # memory_mean = (w_batch @ w_batch^T)^{-1} @ w_batch @ z_batch
                # which is memory_mean = w_batch^+ @ z_batch where w_batch^+ is pseudoinverse
                
                try:
                    # Use PyTorch's pinverse for numerical stability
                    w_pinv = torch.pinverse(w_batch)  # [memory_size, episode_size]
                    memory_mean_batch = torch.mm(w_pinv, z_batch)  # [memory_size, code_size]
                except:
                    # Fallback: use transpose as approximation
                    w_t = w_batch.transpose(0, 1)  # [memory_size, episode_size]
                    memory_mean_batch = torch.mm(w_t, z_batch) / episode_size  # [memory_size, code_size]
                
                new_memory_means.append(memory_mean_batch)
            
            # Stack batch results: [batch_size, memory_size, code_size]
            new_memory_mean = torch.stack(new_memory_means, dim=0)

            posterior_memory = (new_memory_mean, prior_memory[1])
        else:
            # Sequential writing
            memory_state = prior_memory
            episode_size = input_encoded.shape[0]

            for i in range(episode_size):
                z_step = input_encoded[i].unsqueeze(
                    0)  # [1, batch_size, code_size]
                w_step = self.solve_attention_weights(z_step, memory_state[0])
                memory_state = self.update_memory(memory_state, w_step, z_step)

            posterior_memory = memory_state

        # Compute KL divergence (simplified)
        dkl_memory = self.compute_memory_kl(prior_memory, posterior_memory)

        return posterior_memory, dkl_memory

    def read_from_memory(self, query_encoded: torch.Tensor,
                         memory_state: Tuple[torch.Tensor, torch.Tensor],
                         deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read from memory given query
        Args:
            query_encoded: [episode_size, batch_size, code_size]
            memory_state: Current memory state
            deterministic: Whether to use deterministic reading
        Returns:
            retrieved_z: [episode_size, batch_size, code_size]
            attention_weights: [episode_size, batch_size, memory_size]
        """
        memory_mean = self.sample_memory(memory_state)

        # Solve for attention weights
        w = self.solve_attention_weights(
            query_encoded, memory_mean, pseudoinverse=True)

        # Add noise to attention weights if not deterministic
        if not (deterministic or self.deterministic):
            if self.w_logvar_setting == 0:
                w_noise = torch.randn_like(w) * torch.exp(0.5 * self.w_logvar)
            elif self.w_logvar_setting == 1:
                w_std = torch.exp(
                    0.5 * self.w_logvar).unsqueeze(0).unsqueeze(0)
                w_noise = torch.randn_like(w) * w_std
            else:
                w_noise = torch.randn_like(w) * 0.1  # Default noise
            w = w + w_noise

        # Retrieve from memory
        if query_encoded.dim() == 3:
            episode_size, batch_size = query_encoded.shape[:2]
            retrieved_list = []
            for i in range(episode_size):
                w_step = w[i]  # [batch_size, memory_size]
                retrieved_step = torch.bmm(
                    w_step.unsqueeze(1), memory_mean).squeeze(1)
                retrieved_list.append(retrieved_step)
            retrieved_z = torch.stack(retrieved_list, dim=0)
        else:
            # Single timestep
            retrieved_z = torch.bmm(w.unsqueeze(1), memory_mean).squeeze(1)

        return retrieved_z, w

    def approx_pseudoinverse(self, A: torch.Tensor, steps: int = 3) -> torch.Tensor:
        """
        Approximate pseudoinverse using iterative method
        Args:
            A: [batch_size, m, n] matrix
            steps: Number of iteration steps
        Returns:
            A_pinv: Approximate pseudoinverse
        """
        batch_size, m, n = A.shape
        
        # Use Moore-Penrose pseudoinverse approximation
        # Initialize with scaled transpose
        alpha = 2.0 / (torch.norm(A, dim=(1, 2), keepdim=True) ** 2 + 1e-8)
        A_pinv = alpha * A.transpose(1, 2)  # [batch_size, n, m]

        # Iterative refinement using Schulz method
        for _ in range(steps):
            # A_pinv = A_pinv * (2*I - A * A_pinv)
            # Compute A * A_pinv: [batch_size, m, n] @ [batch_size, n, m] = [batch_size, m, m]
            AA_pinv = torch.bmm(A, A_pinv)  # [batch_size, m, m]
            
            # Create identity matrix
            I = torch.eye(m, device=A.device, dtype=A.dtype).unsqueeze(0).expand(batch_size, -1, -1)
            
            # Compute (2*I - A*A_pinv)
            factor = 2 * I - AA_pinv  # [batch_size, m, m]
            
            # Update A_pinv = A_pinv * (2*I - A*A_pinv)
            A_pinv = torch.bmm(A_pinv, factor)  # [batch_size, n, m] @ [batch_size, m, m] = [batch_size, n, m]

        return A_pinv

    def compute_memory_kl(self, prior_memory: Tuple[torch.Tensor, torch.Tensor],
                          posterior_memory: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Compute KL divergence between prior and posterior memory
        Simplified version for computational efficiency
        """
        prior_mean, prior_cov = prior_memory
        posterior_mean, posterior_cov = posterior_memory

        # Simplified KL computation (Frobenius norm of difference)
        mean_diff = posterior_mean - prior_mean
        kl_loss = 0.5 * torch.sum(mean_diff ** 2) / mean_diff.shape[0]

        return kl_loss

    def forward(self, input_encoded: torch.Tensor,
                mode: str = "write") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through memory
        Args:
            input_encoded: [episode_size, batch_size, code_size]
            mode: "write" or "read"
        Returns:
            memory_output: Retrieved or written representations
            memory_state: Updated memory state
        """
        if mode == "write":
            memory_state, kl_loss = self.write_to_memory(input_encoded)
            return memory_state[0], kl_loss  # Return memory mean and KL loss
        elif mode == "read":
            batch_size = input_encoded.shape[1]
            memory_state = self.get_prior_state(batch_size)
            retrieved, attention = self.read_from_memory(
                input_encoded, memory_state)
            return retrieved, attention
        else:
            raise ValueError(f"Unknown mode: {mode}")


class LarimarMemoryVAE(nn.Module):
    """
    Combined memory and VAE module for Tiny-MultiModal-Larimar
    """

    def __init__(self,
                 encoder,
                 decoder,
                 memory_size: int = 512,
                 latent_size: int = 384,
                 memory_direct_writing: bool = True,
                 observation_noise_std: float = 0.1):
        super(LarimarMemoryVAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = latent_size

        # Memory component
        self.memory = TinyLarimarMemory(
            code_size=latent_size,
            memory_size=memory_size,
            direct_writing=memory_direct_writing,
            observation_noise_std=observation_noise_std,
            identity_init=True
        )

        # VAE components
        self.latent_projection = nn.Linear(
            latent_size, latent_size * 2)  # For mu and logvar

        print(f"Initialized LarimarMemoryVAE with memory size {memory_size}")

    def encode(self, input_ids, attention_mask=None, **kwargs):
        """Encode input through encoder"""
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def decode(self, latent_z, input_ids, attention_mask=None, labels=None, **kwargs):
        """Decode from latent space through decoder"""
        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            latent_conditioning=latent_z,
            **kwargs
        )

    def reparameterize(self, mu, logvar):
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input_ids, attention_mask=None, labels=None, use_memory=True, **kwargs):
        """
        Forward pass through the complete model
        """
        # Encode input
        if hasattr(self.encoder, 'encode_to_latent'):
            latent_z, mu, logvar = self.encoder.encode_to_latent(
                input_ids, attention_mask)
        else:
            encoded_output, latent_params = self.encoder(
                input_ids, attention_mask, return_latent_params=True)
            if latent_params is not None:
                mu, logvar = latent_params.chunk(2, -1)
                latent_z = self.reparameterize(mu, logvar)
            else:
                # Fallback if no latent params
                latent_params = self.latent_projection(encoded_output)
                mu, logvar = latent_params.chunk(2, -1)
                latent_z = self.reparameterize(mu, logvar)

        # Memory interaction
        memory_kl = torch.tensor(0.0, device=latent_z.device)
        if use_memory:
            # Prepare for memory (add episode dimension)
            if latent_z.dim() == 2:
                latent_z_mem = latent_z.unsqueeze(
                    0)  # [1, batch_size, latent_size]
            else:
                latent_z_mem = latent_z

            # Write to memory and read back
            memory_state, memory_kl = self.memory.write_to_memory(latent_z_mem)
            retrieved_z, attention_weights = self.memory.read_from_memory(
                latent_z_mem, memory_state)

            # Use retrieved representation
            if retrieved_z.dim() == 3:
                latent_z = retrieved_z.squeeze(0)  # Remove episode dimension
            else:
                latent_z = retrieved_z

        # Decode
        decoder_output = self.decode(
            latent_z, input_ids, attention_mask, labels, **kwargs)

        # Compute losses
        reconstruction_loss = decoder_output.get(
            'loss', torch.tensor(0.0, device=latent_z.device))

        # KL divergence for VAE
        kl_loss = -0.5 * torch.sum(1 + logvar -
                                   mu.pow(2) - logvar.exp()) / mu.shape[0]

        # Total loss
        total_loss = reconstruction_loss + kl_loss + memory_kl

        return {
            'loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_loss,
            'memory_kl': memory_kl,
            'logits': decoder_output.get('logits'),
            'latent_z': latent_z,
            'mu': mu,
            'logvar': logvar
        }
