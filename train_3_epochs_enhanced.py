#!/usr/bin/env python3
"""
Enhanced 3-epoch training with comprehensive W&B logging and visualizations.
Shows embedding learning, multimodal alignment, and model understanding progression.
"""

import sys
sys.path.append("src")

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from tqdm import tqdm
import time

from src.modules.babylm_data import BabyLMMultiModalDataModule
from src.modules.larimar_babylm_lightning import LarimarBabyLMLightningModel
from src.modules.larimar_multimodal_vae import LarimarMultiModalConfig

# Set environment
torch.set_float32_matmul_precision('medium')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MultimodalVisualizationCallback(Callback):
    """Custom callback for comprehensive multimodal learning visualizations"""
    
    def __init__(self, log_every_n_epochs=1, max_samples=300):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.embedding_history = {
            'text_embeddings': [],
            'vision_embeddings': [],
            'multimodal_embeddings': [],
            'memory_states': [],
            'epoch': []
        }
        self.loss_history = {
            'reconstruction': [],
            'kl_text': [],
            'kl_multimodal': [],
            'memory_kl': [],
            'total': [],
            'epoch': []
        }
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log comprehensive visualizations at epoch end"""
        if trainer.current_epoch % self.log_every_n_epochs == 0:
            print(f"\n🎨 Creating visualizations for Epoch {trainer.current_epoch}...")
            
            self.log_embedding_visualizations(trainer, pl_module)
            self.log_multimodal_alignment(trainer, pl_module)
            self.log_memory_utilization(trainer, pl_module)
            self.log_generation_samples(trainer, pl_module)
            self.log_learning_progression(trainer, pl_module)
            self.log_attention_maps(trainer, pl_module)
            
            print(f"✅ Visualizations completed for Epoch {trainer.current_epoch}")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Track loss progression"""
        # Get the latest logged metrics
        if trainer.logged_metrics:
            self.loss_history['reconstruction'].append(
                trainer.logged_metrics.get('train/reconstruction_loss', 0)
            )
            self.loss_history['kl_text'].append(
                trainer.logged_metrics.get('train/text_kl_loss', 0)
            )
            self.loss_history['kl_multimodal'].append(
                trainer.logged_metrics.get('train/multimodal_kl_loss', 0)
            )
            self.loss_history['memory_kl'].append(
                trainer.logged_metrics.get('train/memory_kl_loss', 0)
            )
            self.loss_history['total'].append(
                trainer.logged_metrics.get('train/loss', 0)
            )
            self.loss_history['epoch'].append(trainer.current_epoch)
    
    def log_embedding_visualizations(self, trainer, pl_module):
        """Create t-SNE and PCA visualizations of embeddings"""
        print("📊 Creating embedding visualizations...")
        
        # Get validation dataloader
        val_dataloader = trainer.datamodule.val_dataloader()
        
        # Collect embeddings
        text_embeddings = []
        vision_embeddings = []
        multimodal_embeddings = []
        captions = []
        
        pl_module.eval()
        with torch.no_grad():
            samples_collected = 0
            for batch in val_dataloader:
                if samples_collected >= self.max_samples:
                    break
                
                # Move batch to device
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get embeddings from model
                try:
                    outputs = pl_module.forward(batch)
                    
                    # Extract latent representations
                    if 'text_latent' in outputs:
                        text_embeddings.append(outputs['text_latent'].cpu().numpy())
                    if 'vision_latent' in outputs:
                        vision_embeddings.append(outputs['vision_latent'].cpu().numpy())
                    if 'multimodal_latent' in outputs:
                        multimodal_embeddings.append(outputs['multimodal_latent'].cpu().numpy())
                    
                    # Decode captions for labeling
                    for input_ids in batch['input_ids'][:10]:  # Limit to avoid memory issues
                        caption = pl_module.tokenizer.decode(input_ids, skip_special_tokens=True)
                        captions.append(caption[:30] + "..." if len(caption) > 30 else caption)
                    
                    samples_collected += batch['input_ids'].size(0)
                    
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
        
        if not text_embeddings:
            print("No embeddings collected, skipping visualization")
            return
        
        # Concatenate embeddings
        try:
            text_emb = np.concatenate(text_embeddings, axis=0)
            if vision_embeddings:
                vision_emb = np.concatenate(vision_embeddings, axis=0)
            else:
                vision_emb = text_emb  # Fallback
            if multimodal_embeddings:
                multimodal_emb = np.concatenate(multimodal_embeddings, axis=0)
            else:
                multimodal_emb = text_emb  # Fallback
        except Exception as e:
            print(f"Error concatenating embeddings: {e}")
            return
        
        # Store for history
        self.embedding_history['text_embeddings'].append(text_emb)
        self.embedding_history['vision_embeddings'].append(vision_emb)
        self.embedding_history['multimodal_embeddings'].append(multimodal_emb)
        self.embedding_history['epoch'].append(trainer.current_epoch)
        
        # Create visualizations
        self._create_embedding_plots(text_emb, vision_emb, multimodal_emb, captions, trainer.current_epoch)
    
    def _create_embedding_plots(self, text_emb, vision_emb, multimodal_emb, captions, epoch):
        """Create embedding visualization plots"""
        print("🎯 Creating embedding plots...")
        
        # Sample data for visualization
        n_samples = min(200, len(text_emb))
        indices = np.random.choice(len(text_emb), n_samples, replace=False)
        
        text_sample = text_emb[indices]
        vision_sample = vision_emb[indices]
        multimodal_sample = multimodal_emb[indices]
        captions_sample = [captions[i] for i in indices if i < len(captions)]
        
        # PCA visualization (faster than t-SNE)
        try:
            # Text embeddings PCA
            pca = PCA(n_components=2)
            text_pca = pca.fit_transform(text_sample)
            
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Text embeddings
            scatter1 = ax1.scatter(text_pca[:, 0], text_pca[:, 1], alpha=0.6, c=range(len(text_pca)), cmap='viridis')
            ax1.set_title(f'Text Embeddings PCA (Epoch {epoch})')
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            
            # Vision embeddings PCA
            vision_pca = pca.fit_transform(vision_sample)
            scatter2 = ax2.scatter(vision_pca[:, 0], vision_pca[:, 1], alpha=0.6, c=range(len(vision_pca)), cmap='plasma')
            ax2.set_title(f'Vision Embeddings PCA (Epoch {epoch})')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            
            # Multimodal embeddings PCA
            multimodal_pca = pca.fit_transform(multimodal_sample)
            scatter3 = ax3.scatter(multimodal_pca[:, 0], multimodal_pca[:, 1], alpha=0.6, c=range(len(multimodal_pca)), cmap='coolwarm')
            ax3.set_title(f'Multimodal Embeddings PCA (Epoch {epoch})')
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            
            plt.tight_layout()
            wandb.log({f"embeddings/pca_comparison_epoch_{epoch}": wandb.Image(fig)})
            plt.close()
            
        except Exception as e:
            print(f"Error creating PCA plots: {e}")
        
        # Embedding dimension analysis
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Embedding norms
            text_norms = np.linalg.norm(text_sample, axis=1)
            vision_norms = np.linalg.norm(vision_sample, axis=1)
            multimodal_norms = np.linalg.norm(multimodal_sample, axis=1)
            
            ax1.hist([text_norms, vision_norms, multimodal_norms], 
                    label=['Text', 'Vision', 'Multimodal'], alpha=0.7, bins=30)
            ax1.set_xlabel('L2 Norm')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Embedding Magnitudes (Epoch {epoch})')
            ax1.legend()
            
            # Embedding variance per dimension
            text_var = np.var(text_sample, axis=0)
            ax2.plot(text_var, label='Text', alpha=0.8)
            if vision_sample.shape[1] == text_sample.shape[1]:
                vision_var = np.var(vision_sample, axis=0)
                ax2.plot(vision_var, label='Vision', alpha=0.8)
            multimodal_var = np.var(multimodal_sample, axis=0)
            ax2.plot(multimodal_var, label='Multimodal', alpha=0.8)
            ax2.set_xlabel('Dimension')
            ax2.set_ylabel('Variance')
            ax2.set_title(f'Per-Dimension Variance (Epoch {epoch})')
            ax2.legend()
            
            plt.tight_layout()
            wandb.log({f"embeddings/analysis_epoch_{epoch}": wandb.Image(fig)})
            plt.close()
            
        except Exception as e:
            print(f"Error creating embedding analysis: {e}")
    
    def log_multimodal_alignment(self, trainer, pl_module):
        """Visualize text-vision alignment"""
        print("🔗 Analyzing multimodal alignment...")
        
        val_dataloader = trainer.datamodule.val_dataloader()
        similarities = []
        alignments = []
        
        pl_module.eval()
        with torch.no_grad():
            batch_count = 0
            for batch in val_dataloader:
                if batch_count >= 20:  # Limit for speed
                    break
                
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    outputs = pl_module.forward(batch)
                    
                    if 'text_latent' in outputs and 'vision_latent' in outputs:
                        text_emb = outputs['text_latent']
                        vision_emb = outputs['vision_latent']
                        
                        # Compute cosine similarities
                        similarity = F.cosine_similarity(text_emb, vision_emb, dim=1)
                        similarities.extend(similarity.cpu().numpy())
                        
                        # Compute alignment scores
                        alignment = torch.mean(torch.abs(text_emb - vision_emb), dim=1)
                        alignments.extend(alignment.cpu().numpy())
                    
                    batch_count += 1
                    
                except Exception as e:
                    print(f"Error in alignment computation: {e}")
                    continue
        
        if similarities:
            # Plot similarity and alignment distributions
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Similarity distribution
            ax1.hist(similarities, bins=30, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Cosine Similarity')
            ax1.set_ylabel('Frequency')
            ax1.set_title(f'Text-Vision Similarity (Epoch {trainer.current_epoch})')
            ax1.axvline(np.mean(similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(similarities):.3f}')
            ax1.legend()
            
            # Alignment distribution
            ax2.hist(alignments, bins=30, alpha=0.7, edgecolor='black', color='orange')
            ax2.set_xlabel('L1 Distance')
            ax2.set_ylabel('Frequency')
            ax2.set_title(f'Text-Vision Alignment (Epoch {trainer.current_epoch})')
            ax2.axvline(np.mean(alignments), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(alignments):.3f}')
            ax2.legend()
            
            plt.tight_layout()
            wandb.log({
                f"alignment/distributions_epoch_{trainer.current_epoch}": wandb.Image(fig),
                f"alignment/mean_similarity": np.mean(similarities),
                f"alignment/std_similarity": np.std(similarities),
                f"alignment/mean_l1_distance": np.mean(alignments)
            })
            plt.close()
    
    def log_memory_utilization(self, trainer, pl_module):
        """Visualize memory system utilization"""
        print("🧠 Analyzing memory utilization...")
        
        if hasattr(pl_module.model, 'memory') and pl_module.model.memory is not None:
            try:
                memory = pl_module.model.memory
                
                # Get memory state
                if hasattr(memory, 'code_memory'):
                    memory_state = memory.code_memory.detach().cpu().numpy()
                    
                    # Compute memory utilization metrics
                    memory_norms = np.linalg.norm(memory_state, axis=1)
                    utilization_rate = np.mean(memory_norms > 0.1)  # Threshold for "active"
                    
                    # Create memory visualizations
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # Memory slot utilization
                    ax1.bar(range(len(memory_norms)), memory_norms)
                    ax1.set_xlabel('Memory Slot')
                    ax1.set_ylabel('L2 Norm')
                    ax1.set_title(f'Memory Slot Utilization (Epoch {trainer.current_epoch})')
                    
                    # Memory heatmap
                    subset_size = min(50, memory_state.shape[0])
                    dim_size = min(50, memory_state.shape[1])
                    im = ax2.imshow(memory_state[:subset_size, :dim_size], cmap='viridis', aspect='auto')
                    ax2.set_xlabel('Memory Dimension')
                    ax2.set_ylabel('Memory Slot')
                    ax2.set_title(f'Memory State Heatmap ({subset_size}x{dim_size})')
                    plt.colorbar(im, ax=ax2)
                    
                    # Memory utilization over time
                    if len(self.embedding_history['epoch']) > 1:
                        ax3.plot(self.embedding_history['epoch'], 
                                [utilization_rate] * len(self.embedding_history['epoch']))
                        ax3.set_xlabel('Epoch')
                        ax3.set_ylabel('Utilization Rate')
                        ax3.set_title('Memory Utilization Over Time')
                        ax3.grid(True)
                    
                    # Memory activation distribution
                    ax4.hist(memory_norms, bins=30, alpha=0.7, edgecolor='black')
                    ax4.set_xlabel('Memory Slot Activation')
                    ax4.set_ylabel('Frequency')
                    ax4.set_title('Memory Activation Distribution')
                    ax4.axvline(np.mean(memory_norms), color='red', linestyle='--',
                               label=f'Mean: {np.mean(memory_norms):.3f}')
                    ax4.legend()
                    
                    plt.tight_layout()
                    wandb.log({
                        f"memory/analysis_epoch_{trainer.current_epoch}": wandb.Image(fig),
                        f"memory/utilization_rate": utilization_rate,
                        f"memory/mean_activation": np.mean(memory_norms),
                        f"memory/std_activation": np.std(memory_norms)
                    })
                    plt.close()
                    
            except Exception as e:
                print(f"Error in memory analysis: {e}")
    
    def log_generation_samples(self, trainer, pl_module):
        """Log generated text samples"""
        print("✍️ Generating text samples...")
        
        sample_prompts = [
            "The future of artificial intelligence",
            "In a world where robots and humans",
            "The most beautiful thing about nature",
            "Technology has changed our lives by",
            "Children playing in the park were",
            "Once upon a time in a magical forest",
            "The scientist discovered that",
            "In the year 2050, people will"
        ]
        
        generated_samples = []
        
        pl_module.eval()
        for prompt in sample_prompts:
            try:
                inputs = pl_module.tokenizer(
                    prompt, return_tensors="pt", padding=True, max_length=50
                ).to(pl_module.device)
                
                with torch.no_grad():
                    # Use the model's generate method if available
                    if hasattr(pl_module.model, 'generate'):
                        generated_ids = pl_module.model.generate(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            max_length=80,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9,
                            pad_token_id=pl_module.tokenizer.pad_token_id,
                            eos_token_id=pl_module.tokenizer.eos_token_id
                        )
                        
                        generated_text = pl_module.tokenizer.decode(
                            generated_ids[0], skip_special_tokens=True
                        )
                    else:
                        # Fallback: just return the prompt
                        generated_text = prompt + " [Model does not support generation]"
                    
                    generated_samples.append({
                        "prompt": prompt,
                        "generated": generated_text,
                        "epoch": trainer.current_epoch
                    })
                    
            except Exception as e:
                print(f"Error generating text for prompt '{prompt}': {e}")
                generated_samples.append({
                    "prompt": prompt,
                    "generated": f"[Error: {str(e)}]",
                    "epoch": trainer.current_epoch
                })
        
        # Log as table
        table = wandb.Table(
            columns=["Epoch", "Prompt", "Generated Text"],
            data=[[sample["epoch"], sample["prompt"], sample["generated"]] 
                  for sample in generated_samples]
        )
        
        wandb.log({f"generation/samples_epoch_{trainer.current_epoch}": table})
    
    def log_attention_maps(self, trainer, pl_module):
        """Visualize attention patterns"""
        print("👁️ Analyzing attention patterns...")
        
        # Get a batch for attention analysis
        val_dataloader = trainer.datamodule.val_dataloader()
        
        pl_module.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                try:
                    # Forward pass to get attention if available
                    outputs = pl_module.forward(batch)
                    
                    # Create a simple attention visualization using input representations
                    if 'text_latent' in outputs and 'vision_latent' in outputs:
                        text_emb = outputs['text_latent'][:4]  # First 4 samples
                        vision_emb = outputs['vision_latent'][:4]
                        
                        # Compute attention-like scores
                        attention_scores = torch.matmul(text_emb, vision_emb.transpose(-2, -1))
                        attention_scores = F.softmax(attention_scores, dim=-1)
                        
                        # Plot attention matrix
                        fig, ax = plt.subplots(figsize=(10, 8))
                        im = ax.imshow(attention_scores.cpu().numpy(), cmap='Blues', aspect='auto')
                        ax.set_xlabel('Vision Features')
                        ax.set_ylabel('Text Features')
                        ax.set_title(f'Text-Vision Attention Patterns (Epoch {trainer.current_epoch})')
                        plt.colorbar(im, ax=ax)
                        
                        wandb.log({f"attention/text_vision_epoch_{trainer.current_epoch}": wandb.Image(fig)})
                        plt.close()
                    
                    break  # Only process one batch
                    
                except Exception as e:
                    print(f"Error in attention analysis: {e}")
                    break
    
    def log_learning_progression(self, trainer, pl_module):
        """Track learning progression across epochs"""
        if len(self.loss_history['epoch']) > 1:
            print("📈 Analyzing learning progression...")
            
            # Plot loss curves
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            epochs = self.loss_history['epoch']
            
            # Total loss
            ax1.plot(epochs, self.loss_history['total'], 'b-', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Total Loss')
            ax1.set_title('Total Loss Progression')
            ax1.grid(True)
            
            # Reconstruction loss
            ax2.plot(epochs, self.loss_history['reconstruction'], 'g-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Reconstruction Loss')
            ax2.set_title('Reconstruction Loss Progression')
            ax2.grid(True)
            
            # KL losses
            ax3.plot(epochs, self.loss_history['kl_text'], 'r-', label='Text KL', linewidth=2)
            ax3.plot(epochs, self.loss_history['kl_multimodal'], 'orange', label='Multimodal KL', linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('KL Loss')
            ax3.set_title('KL Loss Progression')
            ax3.legend()
            ax3.grid(True)
            
            # Memory loss
            ax4.plot(epochs, self.loss_history['memory_kl'], 'purple', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Memory KL Loss')
            ax4.set_title('Memory Loss Progression')
            ax4.grid(True)
            
            plt.tight_layout()
            wandb.log({f"progression/loss_curves_epoch_{trainer.current_epoch}": wandb.Image(fig)})
            plt.close()
            
            # Compute embedding drift if we have history
            if len(self.embedding_history['epoch']) > 1:
                current_text = self.embedding_history['text_embeddings'][-1]
                previous_text = self.embedding_history['text_embeddings'][-2]
                
                # Compute embedding drift
                drift = np.mean(np.linalg.norm(current_text - previous_text, axis=1))
                
                wandb.log({
                    f"progression/embedding_drift": drift,
                    f"progression/current_epoch": trainer.current_epoch
                })


def create_enhanced_trainer():
    """Create trainer with comprehensive logging and callbacks"""
    
    # W&B Logger with rich configuration
    wandb_logger = WandbLogger(
        project="tiny-multimodal-larimar-3epochs-enhanced",
        entity="babylm-ntust",
        name=f"enhanced-3epoch-training-{int(time.time())}",
        tags=["3-epochs", "visualizations", "multimodal", "embeddings", "enhanced"],
        log_model=False,  # Disable to save space
        save_code=True
    )
    
    # Custom visualization callback
    viz_callback = MultimodalVisualizationCallback(log_every_n_epochs=1)
    
    # Simple trainer - minimal checkpointing
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=1,
        precision="32-true",  # Use 32-bit for stability
        logger=wandb_logger,
        callbacks=[viz_callback],
        enable_checkpointing=False,  # No checkpoints to save space
        enable_progress_bar=True,
        val_check_interval=0.5,  # Validate twice per epoch
        limit_val_batches=50,   # Limit validation for speed
        log_every_n_steps=100,
        gradient_clip_val=1.0
    )
    
    return trainer, wandb_logger


def main():
    """Main training function with enhanced visualizations"""
    print("🚀 Starting Enhanced 3-Epoch Training with Comprehensive Visualizations")
    print("="*80)
    
    # Create enhanced trainer
    trainer, logger = create_enhanced_trainer()
    
    # Small model config for 3-epoch training
    config = LarimarMultiModalConfig(
        text_model_name="bert-base-uncased",
        decoder_model_name="gpt2",  # Smaller model
        vision_model_name="facebook/dinov2-base",
        text_latent_size=256,        # Smaller latent size
        vision_latent_size=256,
        hidden_size=512,             # Smaller hidden size
        memory_size=128,             # Smaller memory
        max_length=128,              # Shorter sequences
        kl_weight=1.0,
        memory_weight=0.1,
        reconstruction_weight=1.0,
        direct_writing=True,
        identity_init=True,
        observation_noise_std=0.1,
        fusion_type="cross_attention",
        use_cross_attention=True,
        num_attention_heads=8
    )
    
    # Data module
    data_module = BabyLMMultiModalDataModule(
        data_path="data/babylm",
        batch_size=16,
        max_length=128,
        num_workers=2,
        pin_memory=True,
        dataset_type="cc_3M"
    )
    
    # Model
    model = LarimarBabyLMLightningModel(
        config=config,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_steps=500,
        kl_warmup_steps=1000,
        memory_warmup_steps=500,
        max_epochs=3
    )
    
    # Log initial configuration
    logger.experiment.config.update({
        "epochs": 3,
        "model_size": "small",
        "visualization_enabled": True,
        "latent_size": 256,
        "memory_size": 128,
        "batch_size": 16,
        "learning_rate": 1e-4
    })
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 Training for 3 epochs with comprehensive visualizations")
    print("📈 Visualizations will include:")
    print("   • Embedding evolution (PCA, t-SNE)")
    print("   • Multimodal alignment analysis")
    print("   • Memory utilization patterns")
    print("   • Text generation samples")
    print("   • Attention visualizations")
    print("   • Learning progression curves")
    print("="*80)
    
    # Start training
    trainer.fit(model, data_module)
    
    print("\n" + "="*80)
    print("✅ Enhanced 3-Epoch Training Completed!")
    print("="*80)
    print(f"🔗 View results at: https://wandb.ai/babylm-ntust/tiny-multimodal-larimar-3epochs-enhanced")
    print("📊 Check the visualizations to see how the model learned:")
    print("   • Embedding spaces evolution")
    print("   • Multimodal understanding development")
    print("   • Memory system utilization")
    print("   • Text generation quality improvement")
    print("="*80)


if __name__ == "__main__":
    main()
