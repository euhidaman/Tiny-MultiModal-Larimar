#!/usr/bin/env python3

from modules.lightning_model import TinyMultiModalLitModel
from modules.multimodal_vae import TinyMultiModalVAE
import os
import argparse
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor
import json
from typing import Optional, List, Dict, Any
import numpy as np

# Add src to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


class TinyMultiModalInference:
    """
    Inference class for Tiny MultiModal Larimar
    """

    def __init__(self,
                 model_path: str,
                 device: str = "auto",
                 torch_dtype: torch.dtype = torch.float32):
        """
        Initialize inference model
        Args:
            model_path: Path to trained model checkpoint or directory
            device: Device to run inference on
            torch_dtype: Torch data type for inference
        """
        self.device = device if device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        # Load model
        print(f"Loading model from {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()

        # Load tokenizer and image processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "facebook/dinov2-base")

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on {self.device}")

    def _load_model(self, model_path: str) -> TinyMultiModalVAE:
        """Load model from checkpoint or directory"""
        if os.path.isfile(model_path) and model_path.endswith('.ckpt'):
            # Load from Lightning checkpoint
            lit_model = TinyMultiModalLitModel.load_from_checkpoint(model_path)
            model = lit_model.model
        elif os.path.isdir(model_path):
            # Load from saved directory
            model = TinyMultiModalVAE.from_pretrained(model_path)
        else:
            raise ValueError(f"Invalid model path: {model_path}")

        return model.to(self.device).to(self.torch_dtype)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor(image, return_tensors='pt')
        return inputs['pixel_values'].to(self.device).to(self.torch_dtype)

    def preprocess_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Preprocess text for inference"""
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].to(self.device),
            'attention_mask': inputs['attention_mask'].to(self.device)
        }

    def encode_image(self, image_path: str) -> torch.Tensor:
        """Encode image to latent representation"""
        pixel_values = self.preprocess_image(image_path)

        with torch.no_grad():
            mean, logvar = self.model.encode_vision(pixel_values)
            # Use mean for deterministic encoding
            return mean

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to latent representation"""
        text_inputs = self.preprocess_text(text)

        with torch.no_grad():
            mean, logvar = self.model.encode_text(
                text_inputs['input_ids'],
                text_inputs['attention_mask']
            )
            # Use mean for deterministic encoding
            return mean

    def encode_multimodal(self, image_path: str, text: str) -> torch.Tensor:
        """Encode image and text to joint latent representation"""
        pixel_values = self.preprocess_image(image_path)
        text_inputs = self.preprocess_text(text)

        with torch.no_grad():
            mean, logvar = self.model.encode_multimodal(
                pixel_values,
                text_inputs['input_ids'],
                text_inputs['attention_mask']
            )
            # Use mean for deterministic encoding
            return mean

    def generate_caption(self,
                         image_path: str,
                         prompt: str = "",
                         max_length: int = 50,
                         temperature: float = 1.0,
                         top_k: int = 50,
                         top_p: float = 0.95,
                         num_return_sequences: int = 1) -> List[str]:
        """
        Generate caption for image
        Args:
            image_path: Path to image
            prompt: Optional text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            num_return_sequences: Number of captions to generate
        Returns:
            List of generated captions
        """
        pixel_values = self.preprocess_image(image_path)

        # Prepare input
        if prompt:
            text_inputs = self.preprocess_text(prompt)
            input_ids = text_inputs['input_ids']
            attention_mask = text_inputs['attention_mask']
            mode = "multimodal"
        else:
            # Start with BOS token
            input_ids = torch.full(
                (1, 1), self.tokenizer.bos_token_id, device=self.device)
            attention_mask = None
            mode = "vision"

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                mode=mode,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences
            )

        # Decode generated text
        captions = []
        for generated in generated_ids:
            # Remove input tokens if prompt was used
            if prompt:
                generated = generated[input_ids.size(1):]

            # Decode tokens
            caption = self.tokenizer.decode(
                generated, skip_special_tokens=True)
            captions.append(caption.strip())

        return captions

    def generate_from_text(self,
                           text: str,
                           max_length: int = 50,
                           temperature: float = 1.0,
                           top_k: int = 50,
                           top_p: float = 0.95,
                           num_return_sequences: int = 1) -> List[str]:
        """
        Generate text continuation
        Args:
            text: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            num_return_sequences: Number of sequences to generate
        Returns:
            List of generated text
        """
        text_inputs = self.preprocess_text(text)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                mode="text",
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return_sequences
            )

        # Decode generated text
        texts = []
        for generated in generated_ids:
            # Remove input tokens
            generated = generated[text_inputs['input_ids'].size(1):]

            # Decode tokens
            text_output = self.tokenizer.decode(
                generated, skip_special_tokens=True)
            texts.append(text_output.strip())

        return texts

    def compute_similarity(self,
                           image_path: Optional[str] = None,
                           text1: Optional[str] = None,
                           text2: Optional[str] = None,
                           image_path2: Optional[str] = None) -> float:
        """
        Compute similarity between different modalities
        Args:
            image_path: Path to first image
            text1: First text
            text2: Second text (for text-text similarity)
            image_path2: Path to second image (for image-image similarity)
        Returns:
            Cosine similarity score
        """
        embeddings = []

        # Get embeddings
        if image_path:
            embeddings.append(self.encode_image(image_path))

        if text1:
            embeddings.append(self.encode_text(text1))

        if text2:
            embeddings.append(self.encode_text(text2))

        if image_path2:
            embeddings.append(self.encode_image(image_path2))

        if len(embeddings) != 2:
            raise ValueError(
                "Need exactly two inputs for similarity computation")

        # Compute cosine similarity
        emb1, emb2 = embeddings
        similarity = torch.cosine_similarity(emb1, emb2, dim=-1)
        return similarity.item()

    def analyze_memory(self,
                       image_path: Optional[str] = None,
                       text: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze memory state for given input
        Args:
            image_path: Path to image
            text: Input text
        Returns:
            Dictionary with memory analysis
        """
        if not self.model.use_memory:
            return {"error": "Model does not use memory"}

        # Encode input
        if image_path and text:
            latent = self.encode_multimodal(image_path, text)
        elif image_path:
            latent = self.encode_image(image_path)
        elif text:
            latent = self.encode_text(text)
        else:
            raise ValueError("Need either image_path or text")

        # Get memory state
        with torch.no_grad():
            z_retrieved, memory_state, memory_kl = self.model.memory(latent)

        # Compute memory statistics
        memory_analysis = {
            "memory_size": self.model.memory_size,
            "latent_size": self.model.latent_size,
            "memory_kl": memory_kl.mean().item(),
            "memory_utilization": torch.norm(memory_state, dim=-1).mean().item(),
            "retrieval_similarity": torch.cosine_similarity(latent, z_retrieved, dim=-1).item()
        }

        return memory_analysis

    def batch_process(self,
                      inputs: List[Dict[str, Any]],
                      task: str = "caption") -> List[Dict[str, Any]]:
        """
        Process multiple inputs in batch
        Args:
            inputs: List of input dictionaries
            task: Task to perform ("caption", "similarity", "encode")
        Returns:
            List of results
        """
        results = []

        for input_data in inputs:
            try:
                if task == "caption":
                    image_path = input_data.get('image_path')
                    prompt = input_data.get('prompt', '')

                    captions = self.generate_caption(
                        image_path=image_path,
                        prompt=prompt,
                        max_length=input_data.get('max_length', 50),
                        temperature=input_data.get('temperature', 1.0),
                        num_return_sequences=input_data.get(
                            'num_return_sequences', 1)
                    )

                    results.append({
                        'input': input_data,
                        'captions': captions,
                        'status': 'success'
                    })

                elif task == "encode":
                    if 'image_path' in input_data and 'text' in input_data:
                        encoding = self.encode_multimodal(
                            input_data['image_path'], input_data['text'])
                    elif 'image_path' in input_data:
                        encoding = self.encode_image(input_data['image_path'])
                    elif 'text' in input_data:
                        encoding = self.encode_text(input_data['text'])
                    else:
                        raise ValueError("Invalid input for encoding")

                    results.append({
                        'input': input_data,
                        'encoding': encoding.cpu().numpy().tolist(),
                        'status': 'success'
                    })

                else:
                    results.append({
                        'input': input_data,
                        'error': f"Unknown task: {task}",
                        'status': 'error'
                    })

            except Exception as e:
                results.append({
                    'input': input_data,
                    'error': str(e),
                    'status': 'error'
                })

        return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description="Inference with Tiny MultiModal Larimar")

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint or directory')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run inference on')

    # Input
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--text_prompt', type=str, default="",
                        help='Text prompt for generation')
    parser.add_argument('--text_input', type=str, default=None,
                        help='Text input for text-only tasks')

    # Task
    parser.add_argument('--task', type=str, default='caption',
                        choices=['caption', 'generate',
                                 'encode', 'similarity', 'memory'],
                        help='Task to perform')

    # Generation parameters
    parser.add_argument('--max_length', type=int, default=50,
                        help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95,
                        help='Top-p sampling')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help='Number of sequences to generate')

    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='File to save results')
    parser.add_argument('--batch_file', type=str, default=None,
                        help='JSON file with batch inputs')

    args = parser.parse_args()

    # Initialize inference model
    inference = TinyMultiModalInference(
        model_path=args.model_path,
        device=args.device
    )

    # Process batch file if provided
    if args.batch_file:
        with open(args.batch_file, 'r') as f:
            batch_inputs = json.load(f)

        results = inference.batch_process(batch_inputs, task=args.task)

        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(json.dumps(results, indent=2))
        return

    # Single input processing
    results = {}

    if args.task == 'caption':
        if not args.image_path:
            raise ValueError("Image path required for captioning")

        captions = inference.generate_caption(
            image_path=args.image_path,
            prompt=args.text_prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences
        )
        results['captions'] = captions

    elif args.task == 'generate':
        if not args.text_input:
            raise ValueError("Text input required for generation")

        generated_texts = inference.generate_from_text(
            text=args.text_input,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences
        )
        results['generated_texts'] = generated_texts

    elif args.task == 'encode':
        if args.image_path and args.text_input:
            encoding = inference.encode_multimodal(
                args.image_path, args.text_input)
        elif args.image_path:
            encoding = inference.encode_image(args.image_path)
        elif args.text_input:
            encoding = inference.encode_text(args.text_input)
        else:
            raise ValueError(
                "Need either image_path or text_input for encoding")

        results['encoding'] = encoding.cpu().numpy().tolist()

    elif args.task == 'memory':
        memory_analysis = inference.analyze_memory(
            image_path=args.image_path,
            text=args.text_input
        )
        results['memory_analysis'] = memory_analysis

    # Save or print results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
