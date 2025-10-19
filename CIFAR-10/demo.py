#!/usr/bin/env python3
"""
CIFAR-10 Quick Demo Script

This script provides a quick way to get started with CIFAR-10 training.
It demonstrates the basic workflow and can be easily modified for experimentation.
"""

import torch
import matplotlib.pyplot as plt
from data_loader import CIFAR10DataLoader
from train import CIFAR10Trainer
from evaluate import CIFAR10Evaluator

def quick_demo():
    """Run a quick demonstration of the CIFAR-10 project"""
    
    print("=" * 60)
    print("CIFAR-10 QUICK DEMO")
    print("=" * 60)
    
    # Step 1: Load and visualize data
    print("\n1. Loading CIFAR-10 dataset...")
    data_loader = CIFAR10DataLoader(batch_size=64)
    train_loader, test_loader = data_loader.load_data()
    
    print("\n2. Visualizing sample images...")
    data_loader.visualize_samples(num_samples=8)
    
    # Step 2: Train a simple model
    print("\n3. Training a simple CNN model...")
    print("   (This will take a few minutes)")
    
    trainer = CIFAR10Trainer(
        model_name='simple',  # Use simple model for quick demo
        batch_size=64,
        learning_rate=0.001,
        num_epochs=5  # Just 5 epochs for demo
    )
    
    trainer.train()
    
    # Step 3: Evaluate the model
    print("\n4. Evaluating the trained model...")
    try:
        evaluator = CIFAR10Evaluator(
            model_path='./checkpoints/best_model_simple.pth',
            model_name='simple'
        )
        results = evaluator.generate_report()
        
        print(f"\nDemo completed successfully!")
        print(f"Final accuracy: {results['accuracy']*100:.2f}%")
        
    except FileNotFoundError:
        print("Model file not found. Training may have failed.")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)
    print("\nNext steps:")
    print("- Try training with more epochs: modify train.py")
    print("- Experiment with different models: 'cnn', 'resnet'")
    print("- Use TensorBoard to monitor training: tensorboard --logdir=runs")
    print("- Check the README.md for more detailed instructions")

def train_full_model():
    """Train a full model with better settings"""
    
    print("Training full CIFAR-10 model...")
    print("This will take 30-60 minutes depending on your hardware.")
    
    trainer = CIFAR10Trainer(
        model_name='cnn',      # Use standard CNN
        batch_size=128,
        learning_rate=0.001,
        num_epochs=50
    )
    
    trainer.train()
    trainer.plot_training_history()
    
    # Evaluate
    evaluator = CIFAR10Evaluator(
        model_path='./checkpoints/best_model_cnn.pth',
        model_name='cnn'
    )
    evaluator.generate_report()

def compare_models():
    """Compare different model architectures"""
    
    models = ['simple', 'cnn', 'resnet']
    results = {}
    
    for model_name in models:
        print(f"\nTraining {model_name} model...")
        
        trainer = CIFAR10Trainer(
            model_name=model_name,
            batch_size=128,
            learning_rate=0.001,
            num_epochs=10  # Reduced for comparison
        )
        
        trainer.train()
        
        # Quick evaluation
        try:
            evaluator = CIFAR10Evaluator(
                model_path=f'./checkpoints/best_model_{model_name}.pth',
                model_name=model_name
            )
            accuracy, _, _ = evaluator.evaluate_accuracy()
            results[model_name] = accuracy
            print(f"{model_name} accuracy: {accuracy*100:.2f}%")
        except:
            print(f"Could not evaluate {model_name}")
    
    # Print comparison
    print("\nModel Comparison:")
    print("-" * 30)
    for model, acc in results.items():
        print(f"{model:>8}: {acc*100:>6.2f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            quick_demo()
        elif sys.argv[1] == "full":
            train_full_model()
        elif sys.argv[1] == "compare":
            compare_models()
        else:
            print("Usage: python demo.py [demo|full|compare]")
    else:
        print("CIFAR-10 Demo Script")
        print("Usage: python demo.py [demo|full|compare]")
        print("\nOptions:")
        print("  demo    - Quick 5-epoch demo (recommended for first run)")
        print("  full    - Full 50-epoch training")
        print("  compare - Compare different model architectures")
        print("\nRunning quick demo by default...")
        quick_demo()
