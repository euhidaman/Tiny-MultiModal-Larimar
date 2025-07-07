#!/usr/bin/env python3

"""
Evaluation script for Tiny MultiModal Larimar
"""

from inference import TinyMultiModalInference
from modules.data import MultiModalDataModule
from modules.lightning_model import TinyMultiModalLitModel
import os
import argparse
import torch
import lightning as L
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


class TinyMultiModalEvaluator:
    """
    Evaluator for Tiny MultiModal Larimar
    """

    def __init__(self,
                 model_path: str,
                 device: str = "auto"):
        """
        Initialize evaluator
        Args:
            model_path: Path to trained model
            device: Device to use for evaluation
        """
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        print(f"Loading model from {model_path}")
        self.inference = TinyMultiModalInference(
            model_path, device=self.device)
        self.model = self.inference.model

        print("Model loaded successfully")

    def evaluate_reconstruction(self,
                                data_module: MultiModalDataModule,
                                num_samples: int = 100) -> Dict[str, float]:
        """
        Evaluate reconstruction quality
        Args:
            data_module: Data module with test data
            num_samples: Number of samples to evaluate
        Returns:
            Dictionary with reconstruction metrics
        """
        print("Evaluating reconstruction quality...")

        # Get test dataloader
        data_module.setup("test")
        test_loader = data_module.test_dataloader()

        self.model.eval()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_rec_loss = 0.0
        total_memory_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating reconstruction")):
                if num_batches >= num_samples // data_module.batch_size:
                    break

                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Determine mode
                if 'pixel_values' in batch and 'input_ids' in batch:
                    mode = "multimodal"
                elif 'pixel_values' in batch:
                    mode = "vision"
                elif 'input_ids' in batch:
                    mode = "text"
                else:
                    continue

                # Forward pass
                outputs = self.model(
                    pixel_values=batch.get('pixel_values'),
                    input_ids=batch.get('input_ids'),
                    attention_mask=batch.get('attention_mask'),
                    labels=batch.get('labels'),
                    mode=mode
                )

                total_loss += outputs['loss'].item()
                total_kl_loss += outputs['kl_loss'].item()
                total_rec_loss += outputs['reconstruction_loss'].item()
                total_memory_loss += outputs['memory_loss'].item()
                num_batches += 1

        if num_batches == 0:
            return {"error": "No valid batches found"}

        return {
            "total_loss": total_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "reconstruction_loss": total_rec_loss / num_batches,
            "memory_loss": total_memory_loss / num_batches,
            "perplexity": np.exp(total_rec_loss / num_batches)
        }

    def evaluate_generation_quality(self,
                                    prompts: List[str],
                                    num_generations: int = 5) -> Dict[str, Any]:
        """
        Evaluate text generation quality
        Args:
            prompts: List of text prompts
            num_generations: Number of generations per prompt
        Returns:
            Dictionary with generation metrics
        """
        print("Evaluating generation quality...")

        results = []

        for prompt in tqdm(prompts, desc="Generating text"):
            generations = self.inference.generate_from_text(
                text=prompt,
                max_length=50,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                num_return_sequences=num_generations
            )

            results.append({
                "prompt": prompt,
                "generations": generations
            })

        # Compute diversity metrics
        all_generations = []
        for result in results:
            all_generations.extend(result["generations"])

        # Compute unique generations ratio
        unique_generations = len(set(all_generations))
        total_generations = len(all_generations)
        diversity_ratio = unique_generations / \
            total_generations if total_generations > 0 else 0

        # Compute average length
        avg_length = np.mean([len(gen.split()) for gen in all_generations])

        return {
            "num_prompts": len(prompts),
            "total_generations": total_generations,
            "unique_generations": unique_generations,
            "diversity_ratio": diversity_ratio,
            "average_length": avg_length,
            "results": results
        }

    def evaluate_multimodal_alignment(self,
                                      image_caption_pairs: List[Tuple[str, str]],
                                      random_pairs: int = 100) -> Dict[str, float]:
        """
        Evaluate alignment between vision and text modalities
        Args:
            image_caption_pairs: List of (image_path, caption) pairs
            random_pairs: Number of random mismatched pairs to generate
        Returns:
            Dictionary with alignment metrics
        """
        print("Evaluating multimodal alignment...")

        # Compute similarities for matched pairs
        matched_similarities = []
        for image_path, caption in tqdm(image_caption_pairs, desc="Processing matched pairs"):
            try:
                similarity = self.inference.compute_similarity(
                    image_path=image_path,
                    text1=caption
                )
                matched_similarities.append(similarity)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        # Compute similarities for random mismatched pairs
        mismatched_similarities = []
        np.random.seed(42)

        for _ in tqdm(range(random_pairs), desc="Processing mismatched pairs"):
            # Random pairs
            idx1, idx2 = np.random.choice(
                len(image_caption_pairs), 2, replace=False)
            image_path = image_caption_pairs[idx1][0]
            caption = image_caption_pairs[idx2][1]

            try:
                similarity = self.inference.compute_similarity(
                    image_path=image_path,
                    text1=caption
                )
                mismatched_similarities.append(similarity)
            except Exception as e:
                print(f"Error processing mismatched pair: {e}")

        # Compute metrics
        matched_mean = np.mean(matched_similarities)
        matched_std = np.std(matched_similarities)
        mismatched_mean = np.mean(mismatched_similarities)
        mismatched_std = np.std(mismatched_similarities)

        # Compute separation score
        separation_score = (matched_mean - mismatched_mean) / \
            np.sqrt(matched_std**2 + mismatched_std**2)

        return {
            "matched_similarity_mean": matched_mean,
            "matched_similarity_std": matched_std,
            "mismatched_similarity_mean": mismatched_mean,
            "mismatched_similarity_std": mismatched_std,
            "separation_score": separation_score,
            "num_matched_pairs": len(matched_similarities),
            "num_mismatched_pairs": len(mismatched_similarities)
        }

    def evaluate_memory_utilization(self,
                                    data_module: MultiModalDataModule,
                                    num_samples: int = 100) -> Dict[str, Any]:
        """
        Evaluate memory module utilization
        Args:
            data_module: Data module
            num_samples: Number of samples to analyze
        Returns:
            Dictionary with memory metrics
        """
        if not self.model.use_memory:
            return {"error": "Model does not use memory"}

        print("Evaluating memory utilization...")

        # Get test dataloader
        data_module.setup("test")
        test_loader = data_module.test_dataloader()

        memory_stats = []
        memory_states = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Analyzing memory")):
                if len(memory_stats) >= num_samples:
                    break

                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Determine mode
                if 'pixel_values' in batch and 'input_ids' in batch:
                    mode = "multimodal"
                    pixel_values = batch['pixel_values']
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                elif 'pixel_values' in batch:
                    mode = "vision"
                    pixel_values = batch['pixel_values']
                    input_ids = None
                    attention_mask = None
                elif 'input_ids' in batch:
                    mode = "text"
                    pixel_values = None
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                else:
                    continue

                batch_size = pixel_values.size(
                    0) if pixel_values is not None else input_ids.size(0)

                for i in range(batch_size):
                    if len(memory_stats) >= num_samples:
                        break

                    # Get single sample
                    sample_pixel_values = pixel_values[i:i +
                                                       1] if pixel_values is not None else None
                    sample_input_ids = input_ids[i:i +
                                                 1] if input_ids is not None else None
                    sample_attention_mask = attention_mask[i:i +
                                                           1] if attention_mask is not None else None

                    # Encode to latent
                    if mode == "multimodal":
                        mean, logvar = self.model.encode_multimodal(
                            sample_pixel_values, sample_input_ids, sample_attention_mask
                        )
                    elif mode == "vision":
                        mean, logvar = self.model.encode_vision(
                            sample_pixel_values)
                    else:  # text
                        mean, logvar = self.model.encode_text(
                            sample_input_ids, sample_attention_mask)

                    # Process through memory
                    z_retrieved, memory_state, memory_kl = self.model.memory(
                        mean)

                    # Compute statistics
                    memory_utilization = torch.norm(
                        memory_state, dim=-1).mean().item()
                    retrieval_similarity = torch.cosine_similarity(
                        mean, z_retrieved, dim=-1).item()
                    memory_kl_value = memory_kl.mean().item()

                    memory_stats.append({
                        "memory_utilization": memory_utilization,
                        "retrieval_similarity": retrieval_similarity,
                        "memory_kl": memory_kl_value,
                        "mode": mode
                    })

                    # Store memory state for analysis
                    memory_states.append(memory_state.cpu().numpy())

        # Compute overall statistics
        memory_utilizations = [s["memory_utilization"] for s in memory_stats]
        retrieval_similarities = [s["retrieval_similarity"]
                                  for s in memory_stats]
        memory_kls = [s["memory_kl"] for s in memory_stats]

        # Analyze memory state diversity
        if len(memory_states) > 1:
            memory_states_array = np.array(
                memory_states).reshape(len(memory_states), -1)
            # Compute pairwise similarities
            similarities = cosine_similarity(memory_states_array)
            # Average similarity (excluding diagonal)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            avg_memory_similarity = np.mean(similarities[mask])
        else:
            avg_memory_similarity = 0.0

        return {
            "memory_size": self.model.memory_size,
            "num_samples": len(memory_stats),
            "memory_utilization_mean": np.mean(memory_utilizations),
            "memory_utilization_std": np.std(memory_utilizations),
            "retrieval_similarity_mean": np.mean(retrieval_similarities),
            "retrieval_similarity_std": np.std(retrieval_similarities),
            "memory_kl_mean": np.mean(memory_kls),
            "memory_kl_std": np.std(memory_kls),
            "memory_state_similarity": avg_memory_similarity,
            "mode_breakdown": {
                mode: len([s for s in memory_stats if s["mode"] == mode])
                for mode in ["vision", "text", "multimodal"]
            }
        }

    def visualize_latent_space(self,
                               data_module: MultiModalDataModule,
                               num_samples: int = 500,
                               save_path: str = "latent_visualization.png") -> None:
        """
        Visualize latent space using t-SNE
        Args:
            data_module: Data module
            num_samples: Number of samples to visualize
            save_path: Path to save visualization
        """
        print("Visualizing latent space...")

        # Get test dataloader
        data_module.setup("test")
        test_loader = data_module.test_dataloader()

        latent_vectors = []
        modes = []

        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Encoding samples")):
                if len(latent_vectors) >= num_samples:
                    break

                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)

                # Determine mode
                if 'pixel_values' in batch and 'input_ids' in batch:
                    mode = "multimodal"
                    pixel_values = batch['pixel_values']
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                elif 'pixel_values' in batch:
                    mode = "vision"
                    pixel_values = batch['pixel_values']
                    input_ids = None
                    attention_mask = None
                elif 'input_ids' in batch:
                    mode = "text"
                    pixel_values = None
                    input_ids = batch['input_ids']
                    attention_mask = batch['attention_mask']
                else:
                    continue

                batch_size = pixel_values.size(
                    0) if pixel_values is not None else input_ids.size(0)

                for i in range(batch_size):
                    if len(latent_vectors) >= num_samples:
                        break

                    # Get single sample
                    sample_pixel_values = pixel_values[i:i +
                                                       1] if pixel_values is not None else None
                    sample_input_ids = input_ids[i:i +
                                                 1] if input_ids is not None else None
                    sample_attention_mask = attention_mask[i:i +
                                                           1] if attention_mask is not None else None

                    # Encode to latent
                    if mode == "multimodal":
                        mean, _ = self.model.encode_multimodal(
                            sample_pixel_values, sample_input_ids, sample_attention_mask
                        )
                    elif mode == "vision":
                        mean, _ = self.model.encode_vision(sample_pixel_values)
                    else:  # text
                        mean, _ = self.model.encode_text(
                            sample_input_ids, sample_attention_mask)

                    latent_vectors.append(mean.cpu().numpy().flatten())
                    modes.append(mode)

        if len(latent_vectors) < 2:
            print("Not enough samples for visualization")
            return

        # Convert to numpy array
        latent_array = np.array(latent_vectors)

        # Apply t-SNE
        print("Applying t-SNE...")
        tsne = TSNE(n_components=2, random_state=42,
                    perplexity=min(30, len(latent_vectors)-1))
        latent_2d = tsne.fit_transform(latent_array)

        # Create visualization
        plt.figure(figsize=(12, 8))

        # Plot by mode
        mode_colors = {'vision': 'red', 'text': 'blue', 'multimodal': 'green'}
        for mode in set(modes):
            mask = np.array(modes) == mode
            plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                        c=mode_colors.get(mode, 'gray'),
                        label=mode, alpha=0.6, s=20)

        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Latent Space Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Latent space visualization saved to {save_path}")

    def generate_evaluation_report(self,
                                   data_module: MultiModalDataModule,
                                   output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report
        Args:
            data_module: Data module for evaluation
            output_dir: Directory to save results
        Returns:
            Dictionary with all evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        print("Generating comprehensive evaluation report...")

        results = {}

        # 1. Reconstruction evaluation
        print("\n1. Evaluating reconstruction...")
        results["reconstruction"] = self.evaluate_reconstruction(
            data_module, num_samples=100)

        # 2. Generation quality (if we have text data)
        print("\n2. Evaluating generation quality...")
        test_prompts = [
            "The image shows",
            "A person is",
            "In this picture",
            "The scene depicts",
            "This is a photo of"
        ]
        results["generation"] = self.evaluate_generation_quality(
            test_prompts, num_generations=3)

        # 3. Memory utilization (if memory is enabled)
        if self.model.use_memory:
            print("\n3. Evaluating memory utilization...")
            results["memory"] = self.evaluate_memory_utilization(
                data_module, num_samples=100)

        # 4. Latent space visualization
        print("\n4. Creating latent space visualization...")
        viz_path = os.path.join(output_dir, "latent_space.png")
        self.visualize_latent_space(
            data_module, num_samples=200, save_path=viz_path)

        # 5. Model statistics
        print("\n5. Computing model statistics...")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        results["model_stats"] = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            # Assuming float32
            "model_size_mb": total_params * 4 / (1024 * 1024),
            "latent_size": self.model.latent_size,
            "memory_size": self.model.memory_size if self.model.use_memory else 0,
            "use_memory": self.model.use_memory,
            "use_cross_attention": self.model.use_cross_attention
        }

        # Save results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nEvaluation completed! Results saved to {output_dir}")
        print(f"Results summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Model size: {total_params * 4 / (1024 * 1024):.1f} MB")

        if "reconstruction" in results:
            print(
                f"  - Reconstruction loss: {results['reconstruction'].get('reconstruction_loss', 'N/A'):.4f}")
            print(
                f"  - Perplexity: {results['reconstruction'].get('perplexity', 'N/A'):.2f}")

        if "generation" in results:
            print(
                f"  - Generation diversity: {results['generation'].get('diversity_ratio', 'N/A'):.3f}")

        if "memory" in results:
            print(
                f"  - Memory utilization: {results['memory'].get('memory_utilization_mean', 'N/A'):.4f}")

        return results


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(
        description="Evaluate Tiny MultiModal Larimar")

    # Model and data
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint or directory')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to evaluation data')
    parser.add_argument('--data_type', type=str, default='multimodal',
                        choices=['multimodal', 'babylm', 'conceptual'],
                        help='Type of evaluation data')

    # Evaluation parameters
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device for evaluation')

    # Tasks
    parser.add_argument('--tasks', type=str, nargs='+',
                        default=['reconstruction', 'generation',
                                 'memory', 'visualization'],
                        choices=['reconstruction', 'generation',
                                 'memory', 'visualization', 'all'],
                        help='Evaluation tasks to run')

    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = TinyMultiModalEvaluator(
        model_path=args.model_path,
        device=args.device
    )

    # Setup data module
    data_module = MultiModalDataModule(
        train_data_path=args.data_path,  # Use same data for evaluation
        val_data_path=args.data_path,
        test_data_path=args.data_path,
        data_type=args.data_type,
        batch_size=args.batch_size,
        num_workers=2
    )

    # Run evaluation
    if 'all' in args.tasks:
        # Run comprehensive evaluation
        results = evaluator.generate_evaluation_report(
            data_module, args.output_dir)
    else:
        # Run specific tasks
        os.makedirs(args.output_dir, exist_ok=True)
        results = {}

        if 'reconstruction' in args.tasks:
            print("Running reconstruction evaluation...")
            results['reconstruction'] = evaluator.evaluate_reconstruction(
                data_module, args.num_samples
            )

        if 'generation' in args.tasks:
            print("Running generation evaluation...")
            test_prompts = [
                "The image shows",
                "A person is",
                "In this picture",
                "The scene depicts",
                "This is a photo of"
            ]
            results['generation'] = evaluator.evaluate_generation_quality(
                test_prompts, num_generations=3
            )

        if 'memory' in args.tasks:
            print("Running memory evaluation...")
            results['memory'] = evaluator.evaluate_memory_utilization(
                data_module, args.num_samples
            )

        if 'visualization' in args.tasks:
            print("Creating latent space visualization...")
            viz_path = os.path.join(args.output_dir, "latent_space.png")
            evaluator.visualize_latent_space(
                data_module, args.num_samples, viz_path
            )

        # Save results
        results_path = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    print("Evaluation completed!")


if __name__ == "__main__":
    main()
