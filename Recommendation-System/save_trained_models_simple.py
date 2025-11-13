"""Simple script to train and save all models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
from src.models.matrix_factorization import SVDRecommender, ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.deep_learning import NCFRecommender
from src.models.hybrid import HybridRecommender

print("Loading dataset...")
loader = DatasetLoader("movielens")
data_dict = loader.load()
ratings_data = data_dict['ratings']
items_data = data_dict.get('items', None)

print(f"Dataset: {len(ratings_data)} ratings, {ratings_data['user_id'].nunique()} users, {ratings_data['item_id'].nunique()} items")

print("\nPreprocessing...")
preprocessor = DataPreprocessor()
train_data, test_data = preprocessor.train_test_split(ratings_data, test_size=0.2)
train_data = preprocessor.encode_ids(train_data)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

print("\n" + "="*70)
print("TRAINING AND SAVING MODELS")
print("="*70)

# 1. User-based CF
print("\n[1/7] User-based CF...")
user_cf = UserBasedCF(n_neighbors=40, similarity='cosine')
user_cf.fit(train_data)
user_cf.save(output_dir / "user_cf_movielens.pkl")
print("Saved!")

# 2. Item-based CF
print("\n[2/7] Item-based CF...")
item_cf = ItemBasedCF(n_neighbors=40, similarity='cosine')
item_cf.fit(train_data)
item_cf.save(output_dir / "item_cf_movielens.pkl")
print("Saved!")

# 3. SVD
print("\n[3/7] SVD...")
svd = SVDRecommender(n_factors=100, n_epochs=20, lr_all=0.005)
svd.fit(train_data)
svd.save(output_dir / "svd_movielens.pkl")
print("Saved!")

# 4. ALS
print("\n[4/7] ALS...")
als = ALSRecommender(factors=100, regularization=0.01, iterations=15)
als.fit(train_data)
als.save(output_dir / "als_movielens.pkl")
print("Saved!")

# 5. Content-based
print("\n[5/7] Content-based...")
content_based = ContentBasedRecommender(max_features=5000)
content_based.fit(train_data, items_df=items_data)
content_based.save(output_dir / "content_based_movielens.pkl")
print("Saved!")

# 6. NCF
print("\n[6/7] NCF (this will take ~30 seconds)...")
ncf = NCFRecommender(
    embedding_dim=64,
    hidden_layers=[128, 64, 32],
    dropout=0.2,
    learning_rate=0.001,
    batch_size=256,
    epochs=10
)
ncf.fit(train_data)
ncf.save(output_dir / "ncf_movielens.pkl")
print("Saved!")

# 7. Hybrid
print("\n[7/7] Hybrid...")
models_dict = {
    'user_cf': user_cf,
    'item_cf': item_cf,
    'svd': svd,
    'als': als,
    'content_based': content_based,
    'ncf': ncf
}
weights = {
    'user_cf': 0.2,
    'item_cf': 0.2,
    'svd': 0.25,
    'als': 0.25,
    'content_based': 0.05,
    'ncf': 0.05
}
hybrid = HybridRecommender(models=models_dict, weights=weights, fusion_method='weighted')
hybrid.save(output_dir / "hybrid_movielens.pkl")
print("Saved!")

print("\n" + "="*70)
print("ALL MODELS SAVED SUCCESSFULLY!")
print("="*70)
print(f"\nModels saved to: {output_dir.absolute()}")
print("\nSaved models:")
for model_file in sorted(output_dir.glob("*_movielens.pkl")):
    print(f"  - {model_file.name}")
