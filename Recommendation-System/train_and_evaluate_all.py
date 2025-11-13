"""
Comprehensive training and evaluation script for all recommendation models.
Trains models on real-world data and compares their performance.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF
from src.models.matrix_factorization import SVDRecommender, ALSRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.deep_learning import NCFRecommender
from src.models.hybrid import HybridRecommender
from src.evaluation.metrics import Evaluator
from src.utils.logger import setup_logger


def load_and_preprocess_data(dataset_name: str = "movielens") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DataPreprocessor]:
    """
    Load and preprocess dataset.

    Args:
        dataset_name: Name of dataset to load

    Returns:
        Tuple of (train_data, test_data, items_data, preprocessor)
    """
    logger.info(f"Loading dataset: {dataset_name}")

    # Load dataset
    loader = DatasetLoader(dataset_name)
    data_dict = loader.load()
    ratings_data = data_dict['ratings']
    items_data = data_dict.get('items', None)

    logger.info(f"Dataset loaded: {len(ratings_data)} ratings")
    logger.info(f"Users: {ratings_data['user_id'].nunique()}")
    logger.info(f"Items: {ratings_data['item_id'].nunique()}")
    logger.info(f"Sparsity: {1 - len(ratings_data) / (ratings_data['user_id'].nunique() * ratings_data['item_id'].nunique()):.2%}")

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()

    # Split data
    train_data, test_data = preprocessor.train_test_split(
        ratings_data,
        test_size=0.2
    )

    logger.info(f"Train set: {len(train_data)} ratings")
    logger.info(f"Test set: {len(test_data)} ratings")

    # Encode IDs
    train_data = preprocessor.encode_ids(train_data)
    test_data = preprocessor.encode_ids(test_data)

    # Get statistics
    stats = preprocessor.get_statistics(train_data)
    logger.info(f"Statistics: {stats}")

    return train_data, test_data, items_data, preprocessor


def train_collaborative_filtering(train_data: pd.DataFrame) -> Dict:
    """Train collaborative filtering models."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING COLLABORATIVE FILTERING MODELS")
    logger.info("="*70)

    models = {}

    # User-based CF
    logger.info("\n[1/2] Training User-Based Collaborative Filtering...")
    start_time = time.time()
    user_cf = UserBasedCF(n_neighbors=40, similarity='cosine')
    user_cf.fit(train_data)
    train_time = time.time() - start_time
    logger.info(f"✓ User-based CF trained in {train_time:.2f}s")
    models['user_cf'] = {'model': user_cf, 'train_time': train_time}

    # Item-based CF
    logger.info("\n[2/2] Training Item-Based Collaborative Filtering...")
    start_time = time.time()
    item_cf = ItemBasedCF(n_neighbors=40, similarity='cosine')
    item_cf.fit(train_data)
    train_time = time.time() - start_time
    logger.info(f"✓ Item-based CF trained in {train_time:.2f}s")
    models['item_cf'] = {'model': item_cf, 'train_time': train_time}

    return models


def train_matrix_factorization(train_data: pd.DataFrame) -> Dict:
    """Train matrix factorization models."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING MATRIX FACTORIZATION MODELS")
    logger.info("="*70)

    models = {}

    # SVD
    logger.info("\n[1/2] Training SVD (Singular Value Decomposition)...")
    start_time = time.time()
    svd = SVDRecommender(n_factors=100, n_epochs=20, lr_all=0.005)
    svd.fit(train_data)
    train_time = time.time() - start_time
    logger.info(f"✓ SVD trained in {train_time:.2f}s")
    models['svd'] = {'model': svd, 'train_time': train_time}

    # ALS
    logger.info("\n[2/2] Training ALS (Alternating Least Squares)...")
    start_time = time.time()
    als = ALSRecommender(factors=100, regularization=0.01, iterations=15)
    als.fit(train_data)
    train_time = time.time() - start_time
    logger.info(f"✓ ALS trained in {train_time:.2f}s")
    models['als'] = {'model': als, 'train_time': train_time}

    return models


def train_content_based(train_data: pd.DataFrame, items_data: pd.DataFrame) -> Dict:
    """Train content-based recommender."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING CONTENT-BASED RECOMMENDER")
    logger.info("="*70)

    models = {}

    # Check if items have features
    if items_data is not None and 'title' in items_data.columns:
        logger.info("\n[1/1] Training Content-Based Filtering...")
        start_time = time.time()
        content_based = ContentBasedRecommender(max_features=5000)
        content_based.fit(train_data, items_df=items_data)
        train_time = time.time() - start_time
        logger.info(f"✓ Content-based trained in {train_time:.2f}s")
        models['content_based'] = {'model': content_based, 'train_time': train_time}
    else:
        logger.warning("No item features available, skipping content-based model")

    return models


def train_deep_learning(train_data: pd.DataFrame) -> Dict:
    """Train deep learning model."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING DEEP LEARNING MODEL (NCF)")
    logger.info("="*70)

    models = {}

    logger.info("\n[1/1] Training Neural Collaborative Filtering...")
    logger.info("Configuration: 64 embedding dim, [128, 64, 32] hidden layers, 10 epochs")
    start_time = time.time()

    try:
        ncf = NCFRecommender(
            embedding_dim=64,
            hidden_layers=[128, 64, 32],
            dropout=0.2,
            learning_rate=0.001,
            batch_size=256,
            epochs=10  # Reduced for faster training
        )
        ncf.fit(train_data)
        train_time = time.time() - start_time
        logger.info(f"✓ NCF trained in {train_time:.2f}s ({train_time/60:.1f} minutes)")
        models['ncf'] = {'model': ncf, 'train_time': train_time}
    except Exception as e:
        logger.error(f"✗ NCF training failed: {e}")

    return models


def train_hybrid(base_models: Dict) -> Dict:
    """Train hybrid recommender using base models."""
    logger.info("\n" + "="*70)
    logger.info("TRAINING HYBRID RECOMMENDER")
    logger.info("="*70)

    models = {}

    # Extract trained models
    model_dict = {}
    for name, info in base_models.items():
        if 'model' in info:
            model_dict[name] = info['model']

    if len(model_dict) >= 2:
        logger.info(f"\n[1/1] Training Hybrid Model with {len(model_dict)} base models...")
        logger.info(f"Base models: {list(model_dict.keys())}")

        start_time = time.time()

        # Default weights (can be tuned)
        weights = {
            'user_cf': 0.2,
            'item_cf': 0.2,
            'svd': 0.25,
            'als': 0.25,
            'content_based': 0.05,
            'ncf': 0.05
        }

        # Filter weights to only include available models
        weights = {k: v for k, v in weights.items() if k in model_dict}

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}

        logger.info(f"Model weights: {weights}")

        hybrid = HybridRecommender(models=model_dict, weights=weights, fusion_method='weighted')
        # Hybrid doesn't need explicit fit since it uses pre-trained models
        train_time = time.time() - start_time
        logger.info(f"✓ Hybrid model configured in {train_time:.2f}s")
        models['hybrid'] = {'model': hybrid, 'train_time': train_time}
    else:
        logger.warning("Not enough models for hybrid approach (need at least 2)")

    return models


def evaluate_model(model, train_data: pd.DataFrame, test_data: pd.DataFrame,
                   model_name: str, train_time: float) -> Dict:
    """
    Evaluate a single model.

    Args:
        model: Trained model
        train_data: Training data
        test_data: Test data
        model_name: Name of the model
        train_time: Training time in seconds

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"\nEvaluating {model_name}...")

    evaluator = Evaluator(model, train_data, test_data)

    # Compute metrics
    start_time = time.time()

    results = {
        'model_name': model_name,
        'train_time': train_time,
    }

    try:
        # Rating prediction metrics
        logger.info("  Computing RMSE and MAE...")
        rmse = evaluator.rmse()
        mae = evaluator.mae()
        results['rmse'] = rmse
        results['mae'] = mae

        # Ranking metrics for top-10
        k = 10
        logger.info(f"  Computing ranking metrics @{k}...")
        precision = evaluator.precision_at_k(k=k)
        recall = evaluator.recall_at_k(k=k)
        f1 = evaluator.f1_at_k(k=k)
        ndcg = evaluator.ndcg_at_k(k=k)
        map_score = evaluator.map_at_k(k=k)

        results[f'precision@{k}'] = precision
        results[f'recall@{k}'] = recall
        results[f'f1@{k}'] = f1
        results[f'ndcg@{k}'] = ndcg
        results[f'map@{k}'] = map_score

        # Diversity and coverage
        logger.info("  Computing diversity and coverage...")
        diversity = evaluator.diversity_at_k(k=k)
        coverage = evaluator.catalog_coverage(k=k)

        results[f'diversity@{k}'] = diversity
        results['coverage'] = coverage

        eval_time = time.time() - start_time
        results['eval_time'] = eval_time

        logger.info(f"✓ {model_name} evaluated in {eval_time:.2f}s")

    except Exception as e:
        logger.error(f"✗ Error evaluating {model_name}: {e}")
        results['error'] = str(e)

    return results


def print_comparison_table(all_results: List[Dict]):
    """Print comparison table of all models."""
    logger.info("\n" + "="*70)
    logger.info("MODEL PERFORMANCE COMPARISON")
    logger.info("="*70)

    # Create DataFrame for easy formatting
    df = pd.DataFrame(all_results)

    # Select key metrics
    key_metrics = ['model_name', 'rmse', 'mae', 'precision@10', 'recall@10',
                   'ndcg@10', 'diversity@10', 'coverage', 'train_time']

    df_display = df[key_metrics].copy()

    # Format numerical columns
    for col in df_display.columns:
        if col not in ['model_name']:
            if 'time' in col:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}s" if pd.notna(x) else "N/A")
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    # Print table
    print("\n" + df_display.to_string(index=False))

    # Print best models for each metric
    logger.info("\n" + "="*70)
    logger.info("BEST MODELS PER METRIC")
    logger.info("="*70)

    # Rating prediction (lower is better)
    for metric in ['rmse', 'mae']:
        if metric in df.columns:
            best_idx = df[metric].idxmin()
            best_model = df.loc[best_idx, 'model_name']
            best_value = df.loc[best_idx, metric]
            logger.info(f"Best {metric.upper()}: {best_model} ({best_value:.4f})")

    # Ranking metrics (higher is better)
    for metric in ['precision@10', 'recall@10', 'ndcg@10', 'map@10']:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, 'model_name']
            best_value = df.loc[best_idx, metric]
            logger.info(f"Best {metric}: {best_model} ({best_value:.4f})")

    # Diversity and coverage (higher is better)
    for metric in ['diversity@10', 'coverage']:
        if metric in df.columns:
            best_idx = df[metric].idxmax()
            best_model = df.loc[best_idx, 'model_name']
            best_value = df.loc[best_idx, metric]
            logger.info(f"Best {metric}: {best_model} ({best_value:.4f})")


def save_models(all_models: Dict, output_dir: str = "models"):
    """Save all trained models."""
    logger.info("\n" + "="*70)
    logger.info("SAVING TRAINED MODELS")
    logger.info("="*70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, info in all_models.items():
        if 'model' in info:
            model = info['model']
            filepath = output_path / f"{name}_movielens100k.pkl"

            try:
                model.save(filepath)
                logger.info(f"✓ Saved {name} to {filepath}")
            except Exception as e:
                logger.error(f"✗ Failed to save {name}: {e}")


def main():
    """Main training and evaluation pipeline."""
    setup_logger(log_file="logs/training.log")

    logger.info("="*70)
    logger.info("RECOMMENDATION SYSTEM - COMPREHENSIVE TRAINING & EVALUATION")
    logger.info("="*70)

    total_start_time = time.time()

    # 1. Load and preprocess data
    train_data, test_data, items_data, preprocessor = load_and_preprocess_data("movielens")

    # Store all trained models
    all_models = {}

    # 2. Train collaborative filtering models
    cf_models = train_collaborative_filtering(train_data)
    all_models.update(cf_models)

    # 3. Train matrix factorization models
    mf_models = train_matrix_factorization(train_data)
    all_models.update(mf_models)

    # 4. Train content-based model
    cb_models = train_content_based(train_data, items_data)
    all_models.update(cb_models)

    # 5. Train deep learning model
    dl_models = train_deep_learning(train_data)
    all_models.update(dl_models)

    # 6. Train hybrid model
    hybrid_models = train_hybrid(all_models)
    all_models.update(hybrid_models)

    # 7. Evaluate all models
    logger.info("\n" + "="*70)
    logger.info("EVALUATING ALL MODELS")
    logger.info("="*70)

    all_results = []
    for name, info in all_models.items():
        if 'model' in info:
            results = evaluate_model(
                info['model'],
                train_data,
                test_data,
                name,
                info['train_time']
            )
            all_results.append(results)

    # 8. Print comparison
    print_comparison_table(all_results)

    # 9. Save models
    save_models(all_models)

    # 10. Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_path = Path("results") / "model_comparison_movielens100k.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n✓ Results saved to {results_path}")

    total_time = time.time() - total_start_time
    logger.info("\n" + "="*70)
    logger.info(f"TOTAL PIPELINE TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info("="*70)


if __name__ == "__main__":
    main()
