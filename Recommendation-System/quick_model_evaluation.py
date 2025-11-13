"""Quick model evaluation - compare trained models on key metrics."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.models.base import BaseRecommender

print("="*70)
print("QUICK MODEL EVALUATION - MovieLens 100K")
print("="*70)

# Load dataset
print("\nLoading dataset...")
loader = DatasetLoader("movielens")
data_dict = loader.load()
ratings_data = data_dict['ratings']

# Prepare data
print("Preprocessing...")
preprocessor = DataPreprocessor()
train_data, test_data = preprocessor.train_test_split(ratings_data, test_size=0.2)
train_data = preprocessor.encode_ids(train_data)
test_data = preprocessor.encode_ids(test_data)

print(f"Train: {len(train_data)} ratings, Test: {len(test_data)} ratings")

# Load models
models_dir = Path("models")
model_names = ['als', 'svd', 'user_cf', 'item_cf', 'ncf', 'content_based']

models = {}
for name in model_names:
    model_file = models_dir / f"{name}_movielens.pkl"
    if model_file.exists():
        print(f"Loading {name}...")
        models[name] = BaseRecommender.load(str(model_file))

print(f"\nLoaded {len(models)} models: {list(models.keys())}")

# Quick evaluation on small sample
print("\n" + "="*70)
print("RATING PREDICTION ACCURACY (1,000 test samples)")
print("="*70)

# Sample 1000 random test ratings for quick evaluation
test_sample = test_data.sample(min(1000, len(test_data)), random_state=42)

results = []

for model_name, model in models.items():
    print(f"\n[{model_name.upper()}]")

    y_true = []
    y_pred = []
    errors = 0

    for idx, row in test_sample.iterrows():
        try:
            pred = model.predict(row['user_id'], row['item_id'])
            if not np.isnan(pred) and pred > 0:  # Valid prediction
                y_true.append(row['rating'])
                y_pred.append(pred)
        except:
            errors += 1

    if len(y_pred) >= 10:  # Need at least 10 predictions
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        coverage = len(y_pred) / len(test_sample) * 100

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Predictions: {len(y_pred)}/{len(test_sample)}")

        results.append({
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'Coverage': coverage,
            'Predictions': len(y_pred)
        })
    else:
        print(f"  ERROR: Only {len(y_pred)} valid predictions (need at least 10)")

# Summary
print("\n" + "="*70)
print("SUMMARY - RATING PREDICTION")
print("="*70)

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('RMSE')

    print("\nRanked by RMSE (lower is better):")
    print(df.to_string(index=False))

    best_rmse = df.iloc[0]
    best_mae = df.loc[df['MAE'].idxmin()]

    print(f"\nBest RMSE: {best_rmse['Model']} ({best_rmse['RMSE']:.4f})")
    print(f"Best MAE:  {best_mae['Model']} ({best_mae['MAE']:.4f})")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    df.to_csv(results_dir / "quick_evaluation.csv", index=False)
    print(f"\nResults saved to: {results_dir / 'quick_evaluation.csv'}")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print("\nNote: This is a quick evaluation on 1,000 test samples.")
print("For comprehensive evaluation, use evaluate_models.py")
