"""
Inference and generation systems for various tasks
"""

import torch
import torch.nn as nn
from PIL import Image
from typing import List, Dict, Optional, Union
import numpy as np


class ImageCaptioner:
    """
    Image captioning inference system
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        transform,
        device: str = 'cuda',
        max_length: int = 50,
        num_beams: int = 3,
        temperature: float = 1.0
    ):
        """
        Args:
            model: VLM model
            tokenizer: Text tokenizer
            transform: Image transform
            device: Device to run on
            max_length: Maximum caption length
            num_beams: Beam search width
            temperature: Sampling temperature
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device
        self.max_length = max_length
        self.num_beams = num_beams
        self.temperature = temperature

    @torch.no_grad()
    def generate(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        return_attention: bool = False
    ) -> Union[str, Dict]:
        """
        Generate caption for image

        Args:
            image: PIL Image, image path, or tensor
            return_attention: Return attention weights

        Returns:
            Caption string or dict with caption and attention
        """
        # Load and transform image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image = self.transform(image)

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Generate caption
        generated_ids = self.model.generate_caption(
            image,
            max_length=self.max_length,
            num_beams=self.num_beams,
            temperature=self.temperature
        )

        # Decode
        caption = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

        if return_attention:
            # Get attention weights
            outputs = self.model(
                images=image,
                input_ids=generated_ids,
                task='caption',
                return_attention=True
            )

            return {
                'caption': caption,
                'attention_weights': outputs.get('attention_weights')
            }

        return caption

    @torch.no_grad()
    def generate_batch(
        self,
        images: List[Union[Image.Image, str]]
    ) -> List[str]:
        """
        Generate captions for batch of images

        Args:
            images: List of PIL Images or image paths

        Returns:
            List of caption strings
        """
        # Load and transform images
        image_tensors = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            img_tensor = self.transform(img)
            image_tensors.append(img_tensor)

        # Stack into batch
        batch = torch.stack(image_tensors).to(self.device)

        # Generate captions
        generated_ids = self.model.generate_caption(
            batch,
            max_length=self.max_length,
            num_beams=self.num_beams,
            temperature=self.temperature
        )

        # Decode all captions
        captions = []
        for i in range(generated_ids.size(0)):
            caption = self.tokenizer.decode(generated_ids[i].tolist(), skip_special_tokens=True)
            captions.append(caption)

        return captions


class VQASystem:
    """
    Visual Question Answering inference system
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        transform,
        answer_vocab: Dict[int, str],
        device: str = 'cuda'
    ):
        """
        Args:
            model: VLM model
            tokenizer: Text tokenizer
            transform: Image transform
            answer_vocab: Mapping from answer ID to answer text
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.transform = transform
        self.answer_vocab = answer_vocab
        self.device = device

    @torch.no_grad()
    def answer(
        self,
        image: Union[Image.Image, str, torch.Tensor],
        question: str,
        top_k: int = 5
    ) -> Union[str, List[Dict[str, float]]]:
        """
        Answer question about image

        Args:
            image: PIL Image, image path, or tensor
            question: Question text
            top_k: Return top-k answers

        Returns:
            Top answer or list of top-k answers with scores
        """
        # Load and transform image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        if isinstance(image, Image.Image):
            image = self.transform(image)

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Encode question
        encoded = self.tokenizer.encode(question)
        question_ids = torch.tensor([encoded['input_ids']]).to(self.device)
        attention_mask = torch.tensor([encoded['attention_mask']]).to(self.device)

        # Get answer logits
        logits = self.model.answer_question(image, question_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)

        # Get top-k answers
        top_probs, top_indices = torch.topk(probs[0], k=top_k)

        answers = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            answer_text = self.answer_vocab.get(idx, f"answer_{idx}")
            answers.append({
                'answer': answer_text,
                'confidence': prob
            })

        if top_k == 1:
            return answers[0]['answer']

        return answers


class RetrievalSystem:
    """
    Cross-modal retrieval inference system
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        transform,
        device: str = 'cuda'
    ):
        """
        Args:
            model: VLM model
            tokenizer: Text tokenizer
            transform: Image transform
            device: Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device

        # Cache for image and text features
        self.image_cache = {}
        self.text_cache = {}

    @torch.no_grad()
    def index_images(
        self,
        images: List[Union[Image.Image, str]],
        image_ids: Optional[List[str]] = None
    ):
        """
        Index images for retrieval

        Args:
            images: List of PIL Images or image paths
            image_ids: Optional list of image IDs
        """
        if image_ids is None:
            image_ids = [f"image_{i}" for i in range(len(images))]

        for img_id, img in zip(image_ids, images):
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)

            # Get image features
            vision_features, vision_pooled = self.model.vision_encoder(img_tensor)
            self.image_cache[img_id] = vision_pooled.cpu()

    @torch.no_grad()
    def index_texts(
        self,
        texts: List[str],
        text_ids: Optional[List[str]] = None
    ):
        """
        Index texts for retrieval

        Args:
            texts: List of text strings
            text_ids: Optional list of text IDs
        """
        if text_ids is None:
            text_ids = [f"text_{i}" for i in range(len(texts))]

        for txt_id, txt in zip(text_ids, texts):
            encoded = self.tokenizer.encode(txt)
            input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
            attention_mask = torch.tensor([encoded['attention_mask']]).to(self.device)

            # Get text features
            language_features, language_pooled = self.model.language_encoder(
                input_ids, attention_mask
            )
            self.text_cache[txt_id] = language_pooled.cpu()

    @torch.no_grad()
    def search_images_by_text(
        self,
        query_text: str,
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Search images by text query

        Args:
            query_text: Query text
            top_k: Number of results to return

        Returns:
            List of image IDs with similarity scores
        """
        # Encode query
        encoded = self.tokenizer.encode(query_text)
        input_ids = torch.tensor([encoded['input_ids']]).to(self.device)
        attention_mask = torch.tensor([encoded['attention_mask']]).to(self.device)

        language_features, language_pooled = self.model.language_encoder(
            input_ids, attention_mask
        )

        # Compute similarities with all images
        image_ids = list(self.image_cache.keys())
        image_features = torch.stack([self.image_cache[img_id] for img_id in image_ids])
        image_features = image_features.to(self.device)

        # Compute similarity
        similarity = torch.matmul(language_pooled, image_features.t()).squeeze(0)

        # Get top-k
        top_scores, top_indices = torch.topk(similarity, k=min(top_k, len(image_ids)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                'image_id': image_ids[idx],
                'similarity': score
            })

        return results

    @torch.no_grad()
    def search_texts_by_image(
        self,
        query_image: Union[Image.Image, str],
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """
        Search texts by image query

        Args:
            query_image: Query image
            top_k: Number of results to return

        Returns:
            List of text IDs with similarity scores
        """
        # Load and transform image
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')
        img_tensor = self.transform(query_image).unsqueeze(0).to(self.device)

        # Get image features
        vision_features, vision_pooled = self.model.vision_encoder(img_tensor)

        # Compute similarities with all texts
        text_ids = list(self.text_cache.keys())
        text_features = torch.stack([self.text_cache[txt_id] for txt_id in text_ids])
        text_features = text_features.to(self.device)

        # Compute similarity
        similarity = torch.matmul(vision_pooled, text_features.t()).squeeze(0)

        # Get top-k
        top_scores, top_indices = torch.topk(similarity, k=min(top_k, len(text_ids)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                'text_id': text_ids[idx],
                'similarity': score
            })

        return results


if __name__ == "__main__":
    print("Inference module - use with trained VLM model")
    print("Example usage in examples/inference_example.py")
