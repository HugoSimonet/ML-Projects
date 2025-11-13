"""
Quick test script - No data download required
Tests the implementation with synthetic data
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VLMModel
from data.tokenizer import SimpleTokenizer
from data.transforms import get_train_transforms, get_val_transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing"""

    def __init__(self, num_samples=100, transform=None):
        self.num_samples = num_samples
        self.transform = transform or get_val_transforms()
        self.tokenizer = SimpleTokenizer(max_length=20)

        # Build small vocabulary
        sample_captions = [
            "a cat sitting on a chair",
            "a dog running in the park",
            "the sun is shining bright",
            "birds flying in the sky",
            "flowers blooming in spring"
        ]
        self.tokenizer.build_vocab(sample_captions * 20)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create random image
        image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )

        # Random caption
        captions = [
            "a cat sitting on a chair",
            "a dog running in the park",
            "the sun is shining bright",
            "birds flying in the sky",
            "flowers blooming in spring"
        ]
        caption = captions[idx % len(captions)]

        # Transform
        image = self.transform(image)

        # Tokenize
        encoded = self.tokenizer.encode(caption, max_length=20)

        return {
            'image': image,
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'caption': caption,
            'image_id': idx
        }


def test_model_creation():
    """Test model creation"""
    print("\n" + "="*60)
    print("TEST 1: Model Creation")
    print("="*60)

    try:
        model = VLMModel(
            vision_encoder_type='resnet',
            language_encoder_type='custom',
            hidden_dim=256,  # Smaller for testing
            fusion_layers=2,
            pretrained=False  # Don't download pretrained weights
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model created successfully")
        print(f"  Parameters: {num_params:,}")

        return model

    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None


def test_forward_pass(model, device):
    """Test forward pass"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)

    model = model.to(device)
    model.eval()

    try:
        # Create dummy data
        images = torch.randn(2, 3, 224, 224).to(device)
        input_ids = torch.randint(0, 1000, (2, 20)).to(device)
        attention_mask = torch.ones(2, 20).to(device)
        target_ids = torch.randint(0, 1000, (2, 20)).to(device)

        # Test caption generation
        with torch.no_grad():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_ids=target_ids,
                task='caption'
            )

        print(f"✓ Caption forward pass successful")
        print(f"  Output shape: {outputs['caption_logits'].shape}")

        # Test VQA
        with torch.no_grad():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task='vqa'
            )

        print(f"✓ VQA forward pass successful")
        print(f"  Output shape: {outputs['vqa_logits'].shape}")

        # Test retrieval
        with torch.no_grad():
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                task='retrieval'
            )

        print(f"✓ Retrieval forward pass successful")
        print(f"  Similarity matrix shape: {outputs['similarity_matrix'].shape}")

        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop(model, device):
    """Test training loop with synthetic data"""
    print("\n" + "="*60)
    print("TEST 3: Training Loop (2 epochs)")
    print("="*60)

    try:
        # Create synthetic dataset
        train_dataset = SyntheticDataset(num_samples=50, transform=get_train_transforms())
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        # Setup training
        model = model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        print(f"Training on {len(train_dataset)} samples")
        print(f"Batches per epoch: {len(train_loader)}")

        # Training loop
        for epoch in range(2):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/2")

            for batch in progress_bar:
                # Move to device
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Shift targets for teacher forcing
                target_ids = input_ids.clone()

                # Forward pass
                outputs = model(
                    images=images,
                    input_ids=input_ids[:, :-1],
                    attention_mask=attention_mask[:, :-1],
                    target_ids=input_ids[:, :-1],
                    task='caption'
                )

                # Compute loss
                loss = criterion(
                    outputs['caption_logits'].reshape(-1, outputs['caption_logits'].size(-1)),
                    target_ids[:, :-1].reshape(-1)
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(train_loader)
            print(f"✓ Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

        print("\n✓ Training loop completed successfully")
        return True

    except Exception as e:
        print(f"✗ Training loop failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference(model, device):
    """Test inference"""
    print("\n" + "="*60)
    print("TEST 4: Inference (Caption Generation)")
    print("="*60)

    model = model.to(device)
    model.eval()

    try:
        # Create dummy image
        image = torch.randn(1, 3, 224, 224).to(device)

        # Generate caption
        with torch.no_grad():
            generated_ids = model.generate_caption(
                image,
                max_length=20,
                num_beams=1,
                temperature=1.0
            )

        print(f"✓ Caption generation successful")
        print(f"  Generated IDs shape: {generated_ids.shape}")
        print(f"  Generated IDs: {generated_ids[0].tolist()[:10]}...")

        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("""
    ============================================================
       Multi-Modal VLM - Quick Test Suite
    ============================================================
       Testing implementation with synthetic data
       No data download required!
    ============================================================
    """)

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Run tests
    results = {}

    # Test 1: Model Creation
    model = test_model_creation()
    results['model_creation'] = model is not None

    if model is None:
        print("\n✗ Cannot proceed without model")
        return

    # Test 2: Forward Pass
    results['forward_pass'] = test_forward_pass(model, device)

    # Test 3: Training Loop
    results['training_loop'] = test_training_loop(model, device)

    # Test 4: Inference
    results['inference'] = test_inference(model, device)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour implementation is working correctly!")
        print("\nNext steps:")
        print("1. Download real data: python scripts/download_data.py")
        print("2. Prepare data: python scripts/prepare_data.py")
        print("3. Train model: python scripts/train_all.py --task caption --debug")
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")


if __name__ == "__main__":
    main()
