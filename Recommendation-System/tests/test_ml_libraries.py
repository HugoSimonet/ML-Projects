"""Test ML libraries are working correctly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("TESTING ML LIBRARIES")
print("=" * 60)

# Test 1: NumPy
print("\n1. Testing NumPy...")
try:
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    print(f"   NumPy version: {np.__version__}")
    print(f"   Array sum: {arr.sum()}")
    print("   [PASS] NumPy working")
except Exception as e:
    print(f"   [FAIL] NumPy error: {e}")

# Test 2: Pandas
print("\n2. Testing Pandas...")
try:
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    print(f"   Pandas version: {pd.__version__}")
    print(f"   DataFrame shape: {df.shape}")
    print("   [PASS] Pandas working")
except Exception as e:
    print(f"   [FAIL] Pandas error: {e}")

# Test 3: scikit-learn
print("\n3. Testing scikit-learn...")
try:
    from sklearn.metrics import mean_squared_error
    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]
    mse = mean_squared_error(y_true, y_pred)
    print(f"   MSE calculation: {mse:.4f}")
    print("   [PASS] scikit-learn working")
except Exception as e:
    print(f"   [FAIL] scikit-learn error: {e}")

# Test 4: scikit-surprise
print("\n4. Testing scikit-surprise...")
try:
    from surprise import Dataset, SVD
    from surprise.model_selection import cross_validate

    # Load sample data
    data = Dataset.load_builtin('ml-100k', prompt=False)

    # Use SVD
    model = SVD(n_epochs=5, verbose=False)

    # Cross-validate
    results = cross_validate(model, data, measures=['RMSE'], cv=2, verbose=False)
    rmse = results['test_rmse'].mean()

    print(f"   Cross-validation RMSE: {rmse:.4f}")
    print("   [PASS] scikit-surprise working")
except Exception as e:
    print(f"   [FAIL] scikit-surprise error: {e}")

# Test 5: implicit
print("\n5. Testing implicit...")
try:
    from implicit.als import AlternatingLeastSquares
    import scipy.sparse as sp

    # Create small sparse matrix
    data = sp.random(100, 50, density=0.1, format='csr')

    model = AlternatingLeastSquares(factors=10, iterations=5)
    model.fit(data)

    print(f"   Model trained with {model.factors} factors")
    print("   [PASS] implicit working")
except Exception as e:
    print(f"   [FAIL] implicit error: {e}")

# Test 6: PyTorch
print("\n6. Testing PyTorch...")
try:
    import torch

    x = torch.tensor([1.0, 2.0, 3.0])
    y = x * 2

    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Tensor operation: {y.tolist()}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print("   [PASS] PyTorch working")
except Exception as e:
    print(f"   [FAIL] PyTorch error: {e}")

# Test 7: FastAPI
print("\n7. Testing FastAPI...")
try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class Item(BaseModel):
        name: str
        price: float

    @app.get("/")
    def read_root():
        return {"Hello": "World"}

    print("   FastAPI app created successfully")
    print("   [PASS] FastAPI working")
except Exception as e:
    print(f"   [FAIL] FastAPI error: {e}")

print("\n" + "=" * 60)
print("ML LIBRARIES TEST COMPLETE")
print("=" * 60)
