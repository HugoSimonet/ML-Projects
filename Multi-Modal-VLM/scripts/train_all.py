"""
Comprehensive training script for all tasks
"""

import torch
import argparse
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import VLMModel
from data import create_dataloaders, get_tokenizer
from training import Trainer
from utils import set_seed, get_device, count_parameters


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Multi-Modal VLM')

    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--task', type=str, default='caption',
                        choices=['caption', 'vqa', 'retrieval', 'contrastive'],
                        help='Task to train')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with small dataset')

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Create VLM model from config"""
    model = VLMModel(
        vision_encoder_type=config['model']['vision_encoder_type'],
        language_encoder_type=config['model']['language_encoder_type'],
        hidden_dim=config['model']['hidden_dim'],
        fusion_layers=config['model']['fusion_layers'],
        num_heads=config['model']['num_heads'],
        mlp_dim=config['model']['mlp_dim'],
        vocab_size=config['model']['vocab_size'],
        max_length=config['model']['max_length'],
        num_answers=config['model']['num_answers'],
        dropout=config['model']['dropout'],
        use_contrastive=config['model']['use_contrastive'],
        freeze_vision=config['model']['freeze_vision'],
        freeze_language=config['model']['freeze_language'],
        pretrained=config['model']['pretrained']
    )

    return model


def train_caption(config, args):
    """Train image captioning"""
    print("\n" + "="*60)
    print("TRAINING IMAGE CAPTIONING")
    print("="*60)

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Create tokenizer
    print("\nLoading tokenizer...")
    tokenizer_path = Path(args.data_dir) / 'tokenizer.json'
    if tokenizer_path.exists():
        from data.tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer.load(str(tokenizer_path))
        print(f"Loaded tokenizer with vocab size: {len(tokenizer)}")
    else:
        tokenizer = get_tokenizer(
            config['data']['tokenizer_type'],
            max_length=config['data']['max_text_length']
        )
        print("Using default tokenizer")

    # Create dataloaders
    print("\nCreating dataloaders...")
    data_config = config['data']['coco']

    # Use sample data for debug mode
    if args.debug:
        print("DEBUG MODE: Using sample dataset")
        data_config['train_ann'] = str(Path(args.data_dir) / 'sample' / 'captions_sample.json')
        data_config['val_ann'] = data_config['train_ann']
        data_config['data_root'] = str(Path(args.data_dir) / 'coco' / 'images')
        config['training']['batch_size'] = 4
        config['training']['num_epochs'] = 2
        config['training']['logging_steps'] = 10
        config['training']['eval_steps'] = 20

    try:
        train_loader, val_loader = create_dataloaders(
            dataset_name='coco',
            data_root=data_config['data_root'],
            annotation_files={
                'train': data_config['train_ann'],
                'val': data_config['val_ann']
            },
            tokenizer=tokenizer,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"\n⚠ Error creating dataloaders: {e}")
        print("\nPlease ensure:")
        print("1. Data has been downloaded: python scripts/download_data.py")
        print("2. Data has been prepared: python scripts/prepare_data.py")
        return

    # Create trainer
    print("\nCreating trainer...")
    output_dir = Path(args.output_dir) / 'caption'
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task='caption',
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        warmup_steps=config['training']['warmup_steps'],
        scheduler_type=config['training']['scheduler'],
        gradient_clip=config['training']['gradient_clip'],
        accumulation_steps=config['training']['accumulation_steps'],
        device=str(device),
        output_dir=str(output_dir),
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        max_checkpoints=config['training']['max_checkpoints'],
        resume_from=args.resume,
        mixed_precision=config['training']['mixed_precision']
    )

    # Train
    print("\nStarting training...")
    print(f"Total epochs: {config['training']['num_epochs']}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: {len(train_loader) * config['training']['num_epochs']}")

    trainer.train()

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {output_dir}")


def train_vqa(config, args):
    """Train VQA"""
    print("\n" + "="*60)
    print("TRAINING VISUAL QUESTION ANSWERING")
    print("="*60)

    # Set seed
    set_seed(args.seed)

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = create_model(config)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")

    # Create tokenizer
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer(
        config['data']['tokenizer_type'],
        max_length=config['data']['max_text_length']
    )

    # Load answer vocabulary
    import json
    answer_vocab_path = Path(args.data_dir) / 'vqa' / 'answer_vocab.json'
    if answer_vocab_path.exists():
        with open(answer_vocab_path, 'r') as f:
            answer_vocab = json.load(f)
        answer_to_idx = answer_vocab['answer_to_idx']
        print(f"Loaded answer vocabulary: {len(answer_to_idx)} answers")
    else:
        print("⚠ Answer vocabulary not found. Please run: python scripts/prepare_data.py")
        return

    # Create dataloaders
    print("\nCreating dataloaders...")
    data_config = config['data']['vqa']

    try:
        train_loader, val_loader = create_dataloaders(
            dataset_name='vqa',
            data_root=data_config['data_root'],
            annotation_files={
                'train_questions': data_config['train_questions'],
                'train_annotations': data_config['train_annotations'],
                'val_questions': data_config['val_questions'],
                'val_annotations': data_config['val_annotations']
            },
            tokenizer=tokenizer,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            answer_to_idx=answer_to_idx
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"\n⚠ Error creating dataloaders: {e}")
        print("\nPlease ensure VQA data has been downloaded and prepared.")
        return

    # Create trainer
    print("\nCreating trainer...")
    output_dir = Path(args.output_dir) / 'vqa'
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task='vqa',
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        warmup_steps=config['training']['warmup_steps'],
        scheduler_type=config['training']['scheduler'],
        gradient_clip=config['training']['gradient_clip'],
        accumulation_steps=config['training']['accumulation_steps'],
        device=str(device),
        output_dir=str(output_dir),
        logging_steps=config['training']['logging_steps'],
        eval_steps=config['training']['eval_steps'],
        save_steps=config['training']['save_steps'],
        max_checkpoints=config['training']['max_checkpoints'],
        resume_from=args.resume,
        mixed_precision=config['training']['mixed_precision']
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    print("\n✓ Training complete!")
    print(f"Checkpoints saved to: {output_dir}")


def main():
    """Main training function"""
    args = parse_args()

    print("""
    ============================================================
       Multi-Modal VLM Training Script
    ============================================================
    """)

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override config with args
    if args.task:
        config['training']['task'] = args.task
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir

    # Train based on task
    if args.task == 'caption':
        train_caption(config, args)
    elif args.task == 'vqa':
        train_vqa(config, args)
    elif args.task == 'contrastive':
        print("Contrastive training coming soon!")
    elif args.task == 'retrieval':
        print("Retrieval training coming soon!")
    else:
        print(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
