"""Basic tests that don't require heavy ML dependencies."""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that core modules can be imported."""
    try:
        from src.utils import config_loader, logger
        from src.data import preprocessor
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_config_loader():
    """Test configuration loading."""
    from src.utils.config_loader import load_config, get_data_path

    config = load_config()
    assert config is not None
    assert 'datasets' in config
    assert 'models' in config
    assert 'training' in config

    # Test path functions
    data_path = get_data_path('raw')
    assert data_path is not None
    assert isinstance(data_path, Path)


def test_preprocessor_basic():
    """Test basic preprocessor functionality."""
    import pandas as pd
    from src.data.preprocessor import DataPreprocessor

    # Create sample data
    data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3, 3],
        'item_id': [1, 2, 1, 3, 2, 3],
        'rating': [5, 4, 3, 5, 4, 2]
    })

    preprocessor = DataPreprocessor()

    # Test statistics
    stats = preprocessor.get_statistics(data)
    assert stats['n_ratings'] == 6
    assert stats['n_users'] == 3
    assert stats['n_items'] == 3

    # Test train/test split
    train, test = preprocessor.train_test_split(data, test_size=0.3)
    assert len(train) + len(test) <= len(data)
    assert len(train) > 0
    assert len(test) > 0


def test_synthetic_data_creation():
    """Test synthetic data generation."""
    from src.data.dataset_loader import DatasetLoader

    loader = DatasetLoader('synthetic')
    ratings, items = loader.create_synthetic_data(
        n_users=50,
        n_items=30,
        n_ratings=200
    )

    assert len(ratings) > 0
    assert len(items) == 30
    assert 'user_id' in ratings.columns
    assert 'item_id' in ratings.columns
    assert 'rating' in ratings.columns


def test_base_recommender_interface():
    """Test base recommender abstract class."""
    import importlib.util
    import inspect

    # Load the base module directly without importing the __init__ that has dependencies
    spec = importlib.util.spec_from_file_location(
        "base",
        Path(__file__).parent.parent / "src" / "models" / "base.py"
    )
    base_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_module)

    BaseRecommender = base_module.BaseRecommender

    # Check that it's abstract
    assert inspect.isabstract(BaseRecommender)

    # Check required methods exist
    assert hasattr(BaseRecommender, 'fit')
    assert hasattr(BaseRecommender, 'predict')
    assert hasattr(BaseRecommender, 'recommend')
    assert hasattr(BaseRecommender, 'save')
    assert hasattr(BaseRecommender, 'load')


if __name__ == "__main__":
    print("Running basic tests...")

    tests = [
        ("Imports", test_imports),
        ("Config Loader", test_config_loader),
        ("Preprocessor", test_preprocessor_basic),
        ("Synthetic Data", test_synthetic_data_creation),
        ("Base Recommender", test_base_recommender_interface),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
