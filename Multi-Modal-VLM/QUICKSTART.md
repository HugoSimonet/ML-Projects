# Quick Start Guide

## Option 1: Quick Test (No Download Required - 5 minutes)

Test the implementation with synthetic data:

```bash
cd Multi-Modal-VLM
python scripts/quick_test.py
```

This will:
- Create synthetic sample data
- Train for 2 epochs on small dataset
- Test all model components
- Verify implementation works

## Option 2: Full Dataset Training (Requires ~25GB disk space)

### Step 1: Download Data (~20-30 minutes)

```bash
# Download COCO and VQA datasets
python scripts/download_data.py
```

This downloads:
- COCO Captions 2017: ~20GB (118K train images, 5K val images)
- VQA v2.0: ~1GB (questions and annotations)

### Step 2: Prepare Data (~5 minutes)

```bash
# Build vocabularies and verify data
python scripts/prepare_data.py
```

This will:
- Verify downloaded datasets
- Build answer vocabulary for VQA
- Build caption vocabulary
- Create dataset statistics

### Step 3: Train Models

**Image Captioning:**
```bash
python scripts/train_all.py --task caption --debug
```

**Visual Question Answering:**
```bash
python scripts/train_all.py --task vqa
```

**With Custom Config:**
```bash
python scripts/train_all.py --task caption --config configs/my_config.yaml
```

## Training Options

### Debug Mode (Fast Training)
Uses small subset of data (100 images):
```bash
python scripts/train_all.py --task caption --debug
```

### Full Training
```bash
python scripts/train_all.py --task caption --output-dir ./outputs/caption_run1
```

### Resume Training
```bash
python scripts/train_all.py --task caption --resume outputs/caption/checkpoint_epoch_5.pt
```

## Inference

After training, test your model:

```bash
python examples/inference_demo.py
```

Or use programmatically:

```python
from models import VLMModel
from inference import ImageCaptioner
from data import get_tokenizer, get_val_transforms
from PIL import Image
import torch

# Load model
model = VLMModel(vision_encoder_type='resnet')
checkpoint = torch.load('outputs/caption/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create captioner
tokenizer = get_tokenizer('simple')
transform = get_val_transforms()
captioner = ImageCaptioner(model, tokenizer, transform)

# Generate caption
image = Image.open('test_image.jpg')
caption = captioner.generate(image)
print(f"Caption: {caption}")
```

## Evaluation

Evaluate trained model:

```bash
python scripts/evaluate.py --checkpoint outputs/caption/checkpoint_best.pt --task caption
```

## Common Issues

### Issue: Out of Memory
**Solution:** Reduce batch size in config:
```yaml
training:
  batch_size: 16  # Reduce from 32
  accumulation_steps: 2  # Increase to maintain effective batch size
```

### Issue: CUDA Out of Memory
**Solution:** Use CPU or reduce model size:
```bash
python scripts/train_all.py --device cpu --debug
```

### Issue: Data Not Found
**Solution:** Ensure data is downloaded and paths are correct:
```bash
python scripts/prepare_data.py  # Verify data setup
```

## Next Steps

1. **Experiment with architectures:**
   - Try ViT instead of ResNet: `vision_encoder_type: vit`
   - Use pretrained BERT: `language_encoder_type: bert`

2. **Hyperparameter tuning:**
   - Adjust learning rate
   - Change number of fusion layers
   - Modify attention heads

3. **Multi-task training:**
   - Train on multiple tasks simultaneously
   - Use contrastive pretraining

4. **Deploy your model:**
   - Export to ONNX for production
   - Create REST API for inference
   - Build web demo

## Resources

- **Documentation:** See README.md
- **Project Summary:** See PROJECT_SUMMARY.md
- **Configuration:** See configs/default_config.yaml
- **Examples:** See examples/ directory

## Support

For issues and questions:
- Check PROJECT_SUMMARY.md for implementation details
- Review example scripts in examples/
- Check dataset preparation with `python scripts/prepare_data.py`
