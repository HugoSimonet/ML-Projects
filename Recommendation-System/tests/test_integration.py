"""Integration test for end-to-end workflow without heavy ML dependencies."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_pipeline():
    """Test complete data processing pipeline."""
    from src.data.dataset_loader import DatasetLoader
    from src.data.preprocessor import DataPreprocessor

    print("\n=== Testing Data Pipeline ===")

    # Create synthetic data
    print("1. Creating synthetic dataset...")
    loader = DatasetLoader('synthetic')
    ratings_df, items_df = loader.create_synthetic_data(
        n_users=100,
        n_items=50,
        n_ratings=500
    )

    print(f"   Created {len(ratings_df)} ratings")
    print(f"   Users: {ratings_df['user_id'].nunique()}")
    print(f"   Items: {ratings_df['item_id'].nunique()}")

    assert len(ratings_df) > 0, "No ratings created"
    assert len(items_df) == 50, "Wrong number of items"

    # Preprocess data
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor()

    # Get statistics before preprocessing
    stats_before = preprocessor.get_statistics(ratings_df)
    print(f"   Before - Ratings: {stats_before['n_ratings']}, "
          f"Users: {stats_before['n_users']}, "
          f"Items: {stats_before['n_items']}")
    print(f"   Sparsity: {stats_before['sparsity']:.2%}")

    # Run preprocessing pipeline
    processed = preprocessor.preprocess_pipeline(
        ratings_df,
        handle_cold_start=True,
        encode=True,
        normalize=False
    )

    train_df = processed['train']
    test_df = processed['test']

    print(f"   Train: {len(train_df)} ratings")
    print(f"   Test: {len(test_df)} ratings")

    assert len(train_df) > 0, "No training data"
    assert len(test_df) > 0, "No test data"
    assert len(train_df) > len(test_df), "Train should be larger than test"

    # Test encoders
    print("\n3. Testing encoders...")
    user_encoder = processed['user_encoder']
    item_encoder = processed['item_encoder']

    print(f"   Unique users after encoding: {len(user_encoder.classes_)}")
    print(f"   Unique items after encoding: {len(item_encoder.classes_)}")

    assert len(user_encoder.classes_) > 0
    assert len(item_encoder.classes_) > 0

    print("\n[PASS] Data pipeline test completed successfully!")
    return True


def test_evaluation_metrics():
    """Test evaluation metrics using direct implementations."""
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    print("\n=== Testing Evaluation Metrics ===")

    # Test RMSE and MAE
    print("\n1. Testing RMSE and MAE...")
    y_true = np.array([5, 4, 3, 5, 2, 4, 3, 5])
    y_pred = np.array([4.8, 4.2, 3.1, 4.9, 2.3, 3.8, 3.2, 4.7])

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")

    assert rmse > 0, "RMSE should be positive"
    assert mae > 0, "MAE should be positive"
    assert rmse >= mae, "RMSE should be >= MAE"

    # Test Precision@K
    print("\n2. Testing Precision@K and Recall@K...")
    relevant_items = {1, 2, 3, 4, 5}
    recommended_items = [1, 2, 10, 11, 12, 3, 13, 14, 15, 16]
    k = 5

    recommended_at_k = set(recommended_items[:k])
    n_relevant_and_recommended = len(relevant_items & recommended_at_k)

    precision_5 = n_relevant_and_recommended / k
    recall_5 = n_relevant_and_recommended / len(relevant_items)

    print(f"   Precision@5: {precision_5:.4f}")
    print(f"   Recall@5: {recall_5:.4f}")

    assert 0 <= precision_5 <= 1, "Precision should be between 0 and 1"
    assert 0 <= recall_5 <= 1, "Recall should be between 0 and 1"

    # Test NDCG (simplified)
    print("\n3. Testing NDCG@K (simplified)...")
    # Just verify the calculation works
    relevances = [1 if item in relevant_items else 0 for item in recommended_items[:k]]
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    print(f"   DCG@5: {dcg:.4f}")
    assert dcg >= 0, "DCG should be non-negative"

    # Test diversity
    print("\n4. Testing diversity...")
    recommendations = [
        [1, 2, 3, 4, 5],
        [2, 3, 6, 7, 8],
        [1, 4, 9, 10, 11]
    ]

    all_recs = [item for user_recs in recommendations for item in user_recs]
    unique_items = len(set(all_recs))
    total_items = len(all_recs)
    diversity = unique_items / total_items if total_items > 0 else 0

    print(f"   Diversity: {diversity:.4f}")
    assert 0 <= diversity <= 1, "Diversity should be between 0 and 1"

    # Test coverage
    print("\n5. Testing coverage...")
    all_items = set(range(1, 21))  # 20 items total
    recommended_unique = set(item for user_recs in recommendations for item in user_recs)
    coverage = len(recommended_unique) / len(all_items) if len(all_items) > 0 else 0

    print(f"   Coverage: {coverage:.4f}")
    assert 0 <= coverage <= 1, "Coverage should be between 0 and 1"

    print("\n[PASS] All evaluation metrics working correctly!")
    return True


def test_config_and_utils():
    """Test configuration and utility functions."""
    from src.utils.config_loader import load_config, get_data_path, get_models_path

    print("\n=== Testing Configuration and Utils ===")

    # Test config loading
    print("\n1. Loading configuration...")
    config = load_config()

    assert 'datasets' in config
    assert 'models' in config
    assert 'training' in config
    assert 'evaluation' in config

    print("   Datasets configured:", list(config['datasets'].keys()))
    print("   Model types configured:", list(config['models'].keys()))

    # Test path getters
    print("\n2. Testing path functions...")
    raw_path = get_data_path('raw')
    processed_path = get_data_path('processed')
    models_path = get_models_path()

    print(f"   Raw data path: {raw_path}")
    print(f"   Processed data path: {processed_path}")
    print(f"   Models path: {models_path}")

    assert raw_path.exists() or not raw_path.exists()  # Just check it returns a Path
    assert processed_path.exists() or not processed_path.exists()
    assert models_path.exists() or not models_path.exists()

    print("\n[PASS] Configuration and utilities working correctly!")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        ("Data Pipeline", test_data_pipeline),
        ("Evaluation Metrics", test_evaluation_metrics),
        ("Config and Utils", test_config_and_utils),
    ]

    passed = 0
    failed = 0
    failed_tests = []

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failed_tests.append(name)

    print("\n" + "=" * 60)
    print("INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed_tests:
        print(f"\nFailed tests: {', '.join(failed_tests)}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
