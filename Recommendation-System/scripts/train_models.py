"""Script to train and evaluate all recommendation models."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import joblib
from loguru import logger

from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.evaluation.metrics import Evaluator
from src.models.collaborative_filtering import ItemBasedCF, UserBasedCF
from src.models.content_based import ContentBasedRecommender
from src.models.hybrid import HybridRecommender
from src.models.matrix_factorization import ALSRecommender, SVDRecommender
from src.models.deep_learning import NCFRecommender
from src.utils.config_loader import get_models_path
from src.utils.logger import setup_logger
from src.visualization.plots import Plotter


def main():
    """Train and evaluate all models."""
    parser = argparse.ArgumentParser(description="Train recommendation models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="movielens",
        choices=["movielens", "lastfm", "amazon", "synthetic"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="100k",
        help="Dataset version (e.g., 100k for MovieLens)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "user_cf", "item_cf", "svd", "als", "content", "ncf", "hybrid"],
        help="Models to train"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save trained models"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logger(log_file="logs/training.log")

    logger.info("=" * 80)
    logger.info("RECOMMENDATION SYSTEM TRAINING")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading {args.dataset} dataset (version: {args.version})...")
    loader = DatasetLoader(args.dataset, args.version)
    data = loader.load()

    ratings_df = data['ratings']
    items_df = data.get('items')

    logger.info(f"Loaded {len(ratings_df)} ratings")

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(
        ratings_df,
        handle_cold_start=True,
        encode=True,
        normalize=False
    )

    train_df = processed_data['train']
    test_df = processed_data['test']

    logger.info(f"Train: {len(train_df)} ratings, Test: {len(test_df)} ratings")

    # Determine which models to train
    model_choices = args.models
    if "all" in model_choices:
        model_choices = ["user_cf", "item_cf", "svd", "als", "content", "ncf"]

    # Initialize models
    models = {}

    if "user_cf" in model_choices:
        logger.info("Initializing User-based CF...")
        models['user_cf'] = UserBasedCF()

    if "item_cf" in model_choices:
        logger.info("Initializing Item-based CF...")
        models['item_cf'] = ItemBasedCF()

    if "svd" in model_choices:
        logger.info("Initializing SVD...")
        models['svd'] = SVDRecommender()

    if "als" in model_choices:
        logger.info("Initializing ALS...")
        models['als'] = ALSRecommender()

    if "content" in model_choices:
        logger.info("Initializing Content-based...")
        models['content'] = ContentBasedRecommender()

    if "ncf" in model_choices:
        logger.info("Initializing NCF...")
        models['ncf'] = NCFRecommender(epochs=10)  # Reduced epochs for faster training

    # Train models
    logger.info("=" * 80)
    logger.info("TRAINING MODELS")
    logger.info("=" * 80)

    for model_name, model in models.items():
        try:
            logger.info(f"\nTraining {model_name}...")

            if model_name == "content" and items_df is not None:
                # Content-based needs item features
                feature_cols = [col for col in items_df.columns if col != 'item_id'][:3]
                model.fit(train_df, items_df, feature_cols)
            else:
                model.fit(train_df)

            logger.info(f"{model_name} training completed")

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            del models[model_name]

    # Train hybrid model if requested
    if "hybrid" in model_choices and len(models) >= 2:
        logger.info("\nTraining Hybrid model...")
        hybrid_model = HybridRecommender(models)
        try:
            if "content" in models and items_df is not None:
                feature_cols = [col for col in items_df.columns if col != 'item_id'][:3]
                hybrid_model.fit(train_df, items_df, feature_cols)
            else:
                hybrid_model.fit(train_df)
            models['hybrid'] = hybrid_model
            logger.info("Hybrid training completed")
        except Exception as e:
            logger.error(f"Error training hybrid: {e}")

    # Evaluate models
    logger.info("=" * 80)
    logger.info("EVALUATING MODELS")
    logger.info("=" * 80)

    evaluator = Evaluator()
    all_items = set(train_df['item_id'].unique())
    results = {}

    for model_name, model in models.items():
        try:
            logger.info(f"\nEvaluating {model_name}...")
            result = evaluator.evaluate_all(model, test_df, all_items)
            results[model_name] = result
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")

    # Print results summary
    logger.info("=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    for model_name, result in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  RMSE: {result['rating_prediction']['rmse']:.4f}")
        logger.info(f"  MAE: {result['rating_prediction']['mae']:.4f}")
        logger.info(f"  Precision@10: {result['ranking'].get('precision@10', 0):.4f}")
        logger.info(f"  Recall@10: {result['ranking'].get('recall@10', 0):.4f}")
        logger.info(f"  NDCG@10: {result['ranking'].get('ndcg@10', 0):.4f}")
        logger.info(f"  Diversity: {result['diversity_coverage']['diversity']:.4f}")
        logger.info(f"  Coverage: {result['diversity_coverage']['coverage']:.4f}")

    # Save models
    if args.save:
        logger.info("=" * 80)
        logger.info("SAVING MODELS")
        logger.info("=" * 80)

        models_path = get_models_path()
        models_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in models.items():
            try:
                save_path = models_path / f"{model_name}.pkl"
                joblib.dump(model, save_path)
                logger.info(f"Saved {model_name} to {save_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")

    # Visualize results
    if args.visualize and results:
        logger.info("=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)

        plotter = Plotter()

        # Prepare results for visualization
        viz_results = {}
        for model_name, result in results.items():
            viz_results[model_name] = {
                'rmse': result['rating_prediction']['rmse'],
                'mae': result['rating_prediction']['mae'],
                **result['ranking'],
                **result['diversity_coverage']
            }

        # Plot comparisons
        plotter.plot_metrics_comparison(
            viz_results,
            ['rmse', 'mae'],
            save_path='docs/metrics_comparison.png'
        )

        plotter.plot_all_ranking_metrics(
            viz_results,
            [5, 10, 20],
            save_path='docs/ranking_metrics.png'
        )

        plotter.plot_diversity_coverage(
            viz_results,
            save_path='docs/diversity_coverage.png'
        )

        logger.info("Visualizations saved to docs/")

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
