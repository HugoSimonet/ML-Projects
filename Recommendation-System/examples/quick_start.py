"""Quick start example for the recommendation system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.models.collaborative_filtering import ItemBasedCF
from src.models.matrix_factorization import SVDRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import Evaluator
from src.utils.logger import setup_logger

# Setup logging
setup_logger()

print("=" * 80)
print("RECOMMENDATION SYSTEM - QUICK START EXAMPLE")
print("=" * 80)

# 1. Create synthetic data for demonstration
print("\n1. Creating synthetic dataset...")
loader = DatasetLoader('synthetic')
ratings_df, items_df = loader.create_synthetic_data(
    n_users=500,
    n_items=200,
    n_ratings=5000
)
print(f"Created {len(ratings_df)} ratings from {ratings_df['user_id'].nunique()} users "
      f"for {ratings_df['item_id'].nunique()} items")

# 2. Preprocess data
print("\n2. Preprocessing data...")
preprocessor = DataPreprocessor()
processed = preprocessor.preprocess_pipeline(ratings_df)
train_df = processed['train']
test_df = processed['test']
print(f"Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")
print(f"Sparsity: {processed['stats_after']['sparsity']:.2%}")

# 3. Train Item-based Collaborative Filtering
print("\n3. Training Item-based Collaborative Filtering...")
item_cf = ItemBasedCF()
item_cf.fit(train_df)
print("Item-based CF training completed")

# 4. Train SVD
print("\n4. Training SVD Matrix Factorization...")
svd = SVDRecommender(n_epochs=10)
svd.fit(train_df)
print("SVD training completed")

# 5. Create Hybrid Model
print("\n5. Creating Hybrid Model...")
hybrid = HybridRecommender(
    models={'item_cf': item_cf, 'svd': svd},
    weights={'item_cf': 0.5, 'svd': 0.5},
    fusion_method='weighted'
)
hybrid.fit(train_df)
print("Hybrid model created")

# 6. Get recommendations for a sample user
print("\n6. Getting recommendations for user 1...")
user_id = 1
n_recommendations = 10

print("\nItem-based CF recommendations:")
cf_recs = item_cf.recommend(user_id, n_recommendations)
for i, (item_id, score) in enumerate(cf_recs[:5], 1):
    print(f"  {i}. Item {item_id} (score: {score:.4f})")

print("\nSVD recommendations:")
svd_recs = svd.recommend(user_id, n_recommendations)
for i, (item_id, score) in enumerate(svd_recs[:5], 1):
    print(f"  {i}. Item {item_id} (score: {score:.4f})")

print("\nHybrid recommendations:")
hybrid_recs = hybrid.recommend(user_id, n_recommendations)
for i, (item_id, score) in enumerate(hybrid_recs[:5], 1):
    print(f"  {i}. Item {item_id} (score: {score:.4f})")

# 7. Evaluate models
print("\n7. Evaluating models...")
evaluator = Evaluator()
all_items = set(train_df['item_id'].unique())

models = {
    'Item-based CF': item_cf,
    'SVD': svd,
    'Hybrid': hybrid
}

print("\nEvaluation Results:")
print("-" * 80)
print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'Precision@10':<15} {'NDCG@10':<10}")
print("-" * 80)

for model_name, model in models.items():
    results = evaluator.evaluate_all(model, test_df, all_items, k_values=[10])

    rmse = results['rating_prediction']['rmse']
    mae = results['rating_prediction']['mae']
    precision = results['ranking'].get('precision@10', 0)
    ndcg = results['ranking'].get('ndcg@10', 0)

    print(f"{model_name:<20} {rmse:<10.4f} {mae:<10.4f} {precision:<15.4f} {ndcg:<10.4f}")

print("-" * 80)

# 8. Explain hybrid recommendation
print("\n8. Explaining hybrid recommendation for user 1, item 50...")
if 50 in train_df['item_id'].values:
    explanation = hybrid.explain_recommendation(user_id=1, item_id=50)
    print(f"\nFinal prediction: {explanation['final_prediction']:.4f}")
    print("\nModel contributions:")
    for model_name, contrib in explanation['model_contributions'].items():
        print(f"  {model_name}:")
        print(f"    Prediction: {contrib['prediction']:.4f}")
        print(f"    Weight: {contrib['weight']:.2f}")
        print(f"    Contribution: {contrib['contribution']:.4f}")

# 9. Save models
print("\n9. Saving models...")
import joblib
from src.utils.config_loader import get_models_path

models_path = get_models_path()
models_path.mkdir(parents=True, exist_ok=True)

joblib.dump(item_cf, models_path / "item_cf_demo.pkl")
joblib.dump(svd, models_path / "svd_demo.pkl")
joblib.dump(hybrid, models_path / "hybrid_demo.pkl")

print(f"Models saved to {models_path}")

print("\n" + "=" * 80)
print("QUICK START COMPLETED!")
print("=" * 80)
print("\nNext steps:")
print("1. Train on real datasets: python scripts/train_models.py --dataset movielens")
print("2. Start the API: python -m src.api.app")
print("3. Open web interface: web/index.html")
print("4. Run tests: pytest tests/")
