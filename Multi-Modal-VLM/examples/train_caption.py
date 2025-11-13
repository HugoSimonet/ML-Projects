"""
Example: Training image captioning model
"""

import torch
from models import VLMModel
from data import create_dataloaders, get_tokenizer
from training import Trainer


def main():
    # Configuration
    config = {
        'data_root': 'data/coco/images/train2017',
        'train_ann': 'data/coco/annotations/captions_train2017.json',
        'val_ann': 'data/coco/annotations/captions_val2017.json',
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("Initializing model...")
    model = VLMModel(
        vision_encoder_type='resnet',
        language_encoder_type='custom',
        hidden_dim=768,
        fusion_layers=6,
        num_heads=12,
        pretrained=True
    )

    print("Creating tokenizer...")
    tokenizer = get_tokenizer('simple', max_length=77)

    print("Loading datasets...")
    train_loader, val_loader = create_dataloaders(
        dataset_name='coco',
        data_root=config['data_root'],
        annotation_files={
            'train': config['train_ann'],
            'val': config['val_ann']
        },
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        num_workers=4
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        task='caption',
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        warmup_steps=1000,
        device=config['device'],
        output_dir='./outputs/caption',
        logging_steps=100,
        eval_steps=1000,
        save_steps=2000,
        mixed_precision=True
    )

    print("Starting training...")
    trainer.train()

    print("Training complete!")


if __name__ == "__main__":
    main()
