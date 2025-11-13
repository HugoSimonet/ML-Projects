"""Evaluate all trained models and compare their performance."""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.models.base import BaseRecommender
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("="*70)
print("MODEL EVALUATION - MovieLens 100K")
print("="*70)

# Load and prepare data
print("\nLoading dataset...")
loader = DatasetLoader("movielens")
data_dict = loader.load()
ratings_data = data_dict['ratings']

print("Preprocessing...")
preprocessor = DataPreprocessor()
train_data, test_data = preprocessor.train_test_split(ratings_data, test_size=0.2)
train_data = preprocessor.encode_ids(train_data)
test_data = preprocessor.encode_ids(test_data)

print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# Load all trained models
models_dir = Path("models")
model_files = sorted(models_dir.glob("*_movielens.pkl"))

if not model_files:
    print("\nERROR: No trained models found in models/ directory")
    print("Please run save_trained_models_simple.py first")
    sys.exit(1)

print(f"\nFound {len(model_files)} trained models")

models = {}
for model_file in model_files:
    model_name = model_file.stem.replace("_movielens", "")
    print(f"  Loading {model_name}...")
    models[model_name] = BaseRecommender.load(str(model_file))

print("\n" + "="*70)
print("EVALUATING MODELS - RATING PREDICTION")
print("="*70)

results = []

# Take a sample of test data for faster evaluation
test_sample = test_data.sample(min(5000, len(test_data)), random_state=42)
print(f"\nUsing {len(test_sample)} test samples for evaluation")

for model_name, model in models.items():
    print(f"\n[{model_name.upper()}]")
    start_time = time.time()

    # Predict ratings
    y_true = []
    y_pred = []

    errors = 0
    for idx, row in test_sample.iterrows():
        try:
            pred = model.predict(row['user_id'], row['item_id'])
            if not np.isnan(pred):
                y_true.append(row['rating'])
                y_pred.append(pred)
        except:
            errors += 1

    if len(y_pred) > 0:
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        eval_time = time.time() - start_time

        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  Predictions: {len(y_pred)}/{len(test_sample)}")
        print(f"  Errors: {errors}")
        print(f"  Time: {eval_time:.2f}s")

        results.append({
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'predictions': len(y_pred),
            'errors': errors,
            'eval_time': eval_time
        })
    else:
        print(f"  ERROR: No valid predictions")

# Evaluate recommendation quality
print("\n" + "="*70)
print("EVALUATING MODELS - TOP-K RECOMMENDATIONS")
print("="*70)

# Sample users
test_users = test_data['user_id'].unique()[:100]  # Sample 100 users
k_values = [5, 10, 20]

rec_results = []

for model_name, model in models.items():
    print(f"\n[{model_name.upper()}]")

    for k in k_values:
        start_time = time.time()

        precision_scores = []
        recall_scores = []

        for user_id in test_users:
            try:
                # Get ground truth: items user rated highly in test set
                user_test = test_data[test_data['user_id'] == user_id]
                relevant_items = set(user_test[user_test['rating'] >= 4]['item_id'].values)

                if len(relevant_items) == 0:
                    continue

                # Get recommendations
                recs = model.recommend(user_id, n_items=k, exclude_seen=True)
                recommended_items = set([item_id for item_id, score in recs])

                # Calculate metrics
                hits = len(relevant_items & recommended_items)
                precision = hits / k if k > 0 else 0
                recall = hits / len(relevant_items) if len(relevant_items) > 0 else 0

                precision_scores.append(precision)
                recall_scores.append(recall)
            except:
                continue

        if len(precision_scores) > 0:
            avg_precision = np.mean(precision_scores)
            avg_recall = np.mean(recall_scores)
            f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

            eval_time = time.time() - start_time

            print(f"  K={k}: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={f1:.4f} ({eval_time:.1f}s)")

            rec_results.append({
                'model': model_name,
                'k': k,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': f1,
                'eval_time': eval_time
            })

# Create comparison tables
print("\n" + "="*70)
print("RATING PREDICTION COMPARISON")
print("="*70)

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('rmse')

    print("\nRanked by RMSE (lower is better):")
    print(df_results[['model', 'rmse', 'mae', 'eval_time']].to_string(index=False))

    # Find best models
    best_rmse = df_results.iloc[0]
    best_mae = df_results.loc[df_results['mae'].idxmin()]

    print(f"\nBest RMSE: {best_rmse['model']} ({best_rmse['rmse']:.4f})")
    print(f"Best MAE:  {best_mae['model']} ({best_mae['mae']:.4f})")

print("\n" + "="*70)
print("TOP-K RECOMMENDATION COMPARISON")
print("="*70)

if rec_results:
    df_rec = pd.DataFrame(rec_results)

    for k in k_values:
        print(f"\nK={k} Rankings:")
        df_k = df_rec[df_rec['k'] == k].sort_values('f1', ascending=False)
        print(df_k[['model', 'precision', 'recall', 'f1']].to_string(index=False))

        if len(df_k) > 0:
            best = df_k.iloc[0]
            print(f"  Best F1@{k}: {best['model']} ({best['f1']:.4f})")

# Save results
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

if results:
    df_results.to_csv(results_dir / "rating_prediction_results.csv", index=False)
    print(f"\nRating prediction results saved to: {results_dir / 'rating_prediction_results.csv'}")

if rec_results:
    df_rec.to_csv(results_dir / "recommendation_results.csv", index=False)
    print(f"Recommendation results saved to: {results_dir / 'recommendation_results.csv'}")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
