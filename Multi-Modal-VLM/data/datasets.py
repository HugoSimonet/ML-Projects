"""
Dataset implementations for COCO Captions, VQA v2.0, and Flickr30K
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
from typing import List, Dict, Optional, Tuple, Callable
import random
from .transforms import get_train_transforms, get_val_transforms
from .tokenizer import SimpleTokenizer, get_tokenizer


class COCOCaptionDataset(Dataset):
    """
    COCO Captions Dataset
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        tokenizer: Optional[SimpleTokenizer] = None,
        transform: Optional[Callable] = None,
        max_length: int = 77,
        split: str = 'train'
    ):
        """
        Args:
            data_root: Root directory containing images
            annotation_file: Path to COCO annotation JSON file
            tokenizer: Text tokenizer
            transform: Image transforms
            max_length: Maximum caption length
            split: Dataset split ('train' or 'val')
        """
        self.data_root = data_root
        self.split = split
        self.max_length = max_length

        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        # Build image ID to annotations mapping
        self.image_to_captions = {}
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_to_captions:
                self.image_to_captions[image_id] = []
            self.image_to_captions[image_id].append(ann['caption'])

        # Get list of images
        self.images = self.annotations['images']

        # Filter images that have captions
        self.images = [img for img in self.images if img['id'] in self.image_to_captions]

        # Tokenizer
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_length)

        # Transforms
        if transform is None:
            if split == 'train':
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'image', 'input_ids', 'attention_mask', 'caption'
        """
        img_info = self.images[idx]
        image_id = img_info['id']
        image_file = img_info['file_name']

        # Load image
        image_path = os.path.join(self.data_root, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            # Return a dummy black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get random caption for this image
        captions = self.image_to_captions[image_id]
        caption = random.choice(captions)

        # Tokenize caption
        encoded = self.tokenizer.encode(caption, max_length=self.max_length)

        return {
            'image': image,
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'caption': caption,
            'image_id': image_id
        }


class VQADataset(Dataset):
    """
    VQA v2.0 Dataset
    """

    def __init__(
        self,
        data_root: str,
        questions_file: str,
        annotations_file: Optional[str] = None,
        tokenizer: Optional[SimpleTokenizer] = None,
        transform: Optional[Callable] = None,
        max_length: int = 77,
        split: str = 'train',
        answer_to_idx: Optional[Dict[str, int]] = None,
        top_k_answers: int = 3129
    ):
        """
        Args:
            data_root: Root directory containing images
            questions_file: Path to VQA questions JSON file
            annotations_file: Path to VQA annotations JSON file (None for test)
            tokenizer: Text tokenizer
            transform: Image transforms
            max_length: Maximum question length
            split: Dataset split
            answer_to_idx: Answer to index mapping
            top_k_answers: Number of top answers to consider
        """
        self.data_root = data_root
        self.split = split
        self.max_length = max_length
        self.top_k_answers = top_k_answers

        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        self.questions = questions_data['questions']

        # Load annotations (if available)
        self.annotations = None
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r') as f:
                annotations_data = json.load(f)
            self.annotations = {ann['question_id']: ann for ann in annotations_data['annotations']}

        # Build answer vocabulary if not provided
        if answer_to_idx is None and self.annotations is not None:
            self.answer_to_idx = self._build_answer_vocab()
        else:
            self.answer_to_idx = answer_to_idx or {}

        # Tokenizer
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_length)

        # Transforms
        if transform is None:
            if split == 'train':
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_transforms()
        else:
            self.transform = transform

    def _build_answer_vocab(self) -> Dict[str, int]:
        """Build vocabulary from top-K answers"""
        answer_freq = {}

        for ann in self.annotations.values():
            for answer_info in ann['answers']:
                answer = answer_info['answer'].lower()
                answer_freq[answer] = answer_freq.get(answer, 0) + 1

        # Sort by frequency
        sorted_answers = sorted(answer_freq.items(), key=lambda x: x[1], reverse=True)

        # Take top-K answers
        answer_to_idx = {}
        for i, (answer, _) in enumerate(sorted_answers[:self.top_k_answers]):
            answer_to_idx[answer] = i

        return answer_to_idx

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'image', 'input_ids', 'attention_mask', 'question', 'answer', 'answer_label'
        """
        question_info = self.questions[idx]
        question_id = question_info['question_id']
        question_text = question_info['question']
        image_id = question_info['image_id']

        # Load image
        image_file = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.data_root, image_file)

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Tokenize question
        encoded = self.tokenizer.encode(question_text, max_length=self.max_length)

        result = {
            'image': image,
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'question': question_text,
            'question_id': question_id
        }

        # Add answer if available
        if self.annotations and question_id in self.annotations:
            ann = self.annotations[question_id]
            answers = [a['answer'].lower() for a in ann['answers']]
            most_common_answer = max(set(answers), key=answers.count)

            answer_label = self.answer_to_idx.get(most_common_answer, -1)

            result['answer'] = most_common_answer
            result['answer_label'] = torch.tensor(answer_label, dtype=torch.long)

        return result


class Flickr30KDataset(Dataset):
    """
    Flickr30K Dataset
    """

    def __init__(
        self,
        data_root: str,
        annotation_file: str,
        tokenizer: Optional[SimpleTokenizer] = None,
        transform: Optional[Callable] = None,
        max_length: int = 77,
        split: str = 'train'
    ):
        """
        Args:
            data_root: Root directory containing images
            annotation_file: Path to annotation file (image_id | caption format)
            tokenizer: Text tokenizer
            transform: Image transforms
            max_length: Maximum caption length
            split: Dataset split
        """
        self.data_root = data_root
        self.split = split
        self.max_length = max_length

        # Load annotations
        self.data = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    image_file = parts[0]
                    caption = parts[1]
                    self.data.append({'image_file': image_file, 'caption': caption})

        # Tokenizer
        self.tokenizer = tokenizer or SimpleTokenizer(max_length=max_length)

        # Transforms
        if transform is None:
            if split == 'train':
                self.transform = get_train_transforms()
            else:
                self.transform = get_val_transforms()
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'image', 'input_ids', 'attention_mask', 'caption'
        """
        item = self.data[idx]
        image_file = item['image_file']
        caption = item['caption']

        # Load image
        image_path = os.path.join(self.data_root, image_file)
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        encoded = self.tokenizer.encode(caption, max_length=self.max_length)

        return {
            'image': image,
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'caption': caption
        }


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning with image-text pairs
    """

    def __init__(
        self,
        base_dataset: Dataset,
        negative_samples: int = 1
    ):
        """
        Args:
            base_dataset: Base dataset (COCO or Flickr30K)
            negative_samples: Number of negative text samples per image
        """
        self.base_dataset = base_dataset
        self.negative_samples = negative_samples

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'image', 'positive_text', 'negative_texts'
        """
        # Get positive pair
        item = self.base_dataset[idx]

        # Get negative samples
        negative_indices = random.sample(
            [i for i in range(len(self.base_dataset)) if i != idx],
            self.negative_samples
        )

        negative_items = [self.base_dataset[i] for i in negative_indices]

        result = {
            'image': item['image'],
            'positive_input_ids': item['input_ids'],
            'positive_attention_mask': item['attention_mask'],
            'negative_input_ids': torch.stack([neg['input_ids'] for neg in negative_items]),
            'negative_attention_mask': torch.stack([neg['attention_mask'] for neg in negative_items])
        }

        return result


def create_dataloaders(
    dataset_name: str,
    data_root: str,
    annotation_files: Dict[str, str],
    tokenizer: Optional[SimpleTokenizer] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders

    Args:
        dataset_name: Name of dataset ('coco', 'vqa', 'flickr30k')
        data_root: Root directory for images
        annotation_files: Dict with 'train' and 'val' annotation file paths
        tokenizer: Text tokenizer
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional dataset arguments

    Returns:
        train_loader, val_loader
    """
    if dataset_name == 'coco':
        train_dataset = COCOCaptionDataset(
            data_root=data_root,
            annotation_file=annotation_files['train'],
            tokenizer=tokenizer,
            split='train',
            **kwargs
        )
        val_dataset = COCOCaptionDataset(
            data_root=data_root,
            annotation_file=annotation_files['val'],
            tokenizer=tokenizer,
            split='val',
            **kwargs
        )

    elif dataset_name == 'vqa':
        train_dataset = VQADataset(
            data_root=data_root,
            questions_file=annotation_files['train_questions'],
            annotations_file=annotation_files.get('train_annotations'),
            tokenizer=tokenizer,
            split='train',
            **kwargs
        )
        val_dataset = VQADataset(
            data_root=data_root,
            questions_file=annotation_files['val_questions'],
            annotations_file=annotation_files.get('val_annotations'),
            tokenizer=tokenizer,
            split='val',
            **kwargs
        )

    elif dataset_name == 'flickr30k':
        train_dataset = Flickr30KDataset(
            data_root=data_root,
            annotation_file=annotation_files['train'],
            tokenizer=tokenizer,
            split='train',
            **kwargs
        )
        val_dataset = Flickr30KDataset(
            data_root=data_root,
            annotation_file=annotation_files['val'],
            tokenizer=tokenizer,
            split='val',
            **kwargs
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = SimpleTokenizer()

    print("Testing dataset creation...")
    print("Note: This requires actual COCO/VQA/Flickr30K data to run")
    print("Example usage provided in examples directory")
