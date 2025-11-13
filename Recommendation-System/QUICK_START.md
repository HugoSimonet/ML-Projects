# Quick Start Guide - Python 3.12 Setup

## ✅ Verified Working with Your System!

You already have Python 3.12 installed, which works perfectly with all ML libraries.

---

## Step-by-Step Setup (5 minutes)

### 1. Open Command Prompt or PowerShell

Navigate to your project:
```bash
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System
```

### 2. Create Virtual Environment with Python 3.12
```bash
py -3.12 -m venv venv312
```

### 3. Activate the Virtual Environment
```bash
venv312\Scripts\activate
```

You should see `(venv312)` at the beginning of your command prompt.

### 4. Install All Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- NumPy (v1.x - compatible with surprise)
- pandas, scipy, scikit-learn
- scikit-surprise ✓ (Collaborative Filtering)
- implicit ✓ (ALS Matrix Factorization)
- PyTorch ✓ (Deep Learning)
- FastAPI ✓ (API Server)
- And all other dependencies

**This may take 5-10 minutes** depending on your internet speed.

### 5. Install Project
```bash
pip install -e .
```

---

## Verify Installation

Run the test suite:
```bash
python tests/test_basic.py
python tests/test_integration.py
```

Expected output: All tests passing ✓

---

## Run Your First Example

```bash
python examples/quick_start.py
```

This will:
1. Create synthetic data (500 users, 200 items, 5000 ratings)
2. Train 3 models (ItemCF, SVD, Hybrid)
3. Generate recommendations
4. Evaluate performance
5. Save trained models

**Expected runtime**: 2-3 minutes

---

## Train All Models

To train all 6 recommendation algorithms:

```bash
python scripts/train_models.py --dataset synthetic --models all --save --visualize
```

This trains:
- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- SVD Matrix Factorization
- ALS Matrix Factorization
- Content-Based Filtering
- Neural Collaborative Filtering (Deep Learning)
- Hybrid System (combines all models)

**Expected runtime**: 5-10 minutes

---

## Start the API Server

```bash
python -m src.api.app
```

API will be available at:
- **API**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs

Then open `web/index.html` in your browser to use the demo interface!

---

## Common Commands

### Activate Environment
```bash
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System
venv312\Scripts\activate
```

### Deactivate Environment
```bash
deactivate
```

### Run Tests
```bash
pytest tests/ -v
```

### Train on MovieLens
```bash
python scripts/train_models.py --dataset movielens --version 100k --models all --save
```

### Check Installed Packages
```bash
pip list
```

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

---

## What You Can Do Now

✅ **Data Processing**
```python
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor

loader = DatasetLoader('synthetic')
ratings, items = loader.create_synthetic_data(n_users=1000, n_items=500)

preprocessor = DataPreprocessor()
processed = preprocessor.preprocess_pipeline(ratings)
```

✅ **Train Models**
```python
from src.models.collaborative_filtering import ItemBasedCF
from src.models.matrix_factorization import SVDRecommender

model = SVDRecommender()
model.fit(train_data)
recommendations = model.recommend(user_id=1, n_items=10)
```

✅ **Evaluate**
```python
from src.evaluation.metrics import Evaluator

evaluator = Evaluator()
results = evaluator.evaluate_all(model, test_data, all_items)
```

✅ **API**
```python
import requests

response = requests.post('http://localhost:8000/recommend', json={
    'user_id': 1,
    'n_items': 10
})
print(response.json())
```

---

## Troubleshooting

### Issue: "py -3.12" not found
Try: `python -m venv venv312` (uses default Python)

### Issue: Installation fails
```bash
# Update pip first
python -m pip install --upgrade pip

# Then try again
pip install -r requirements.txt
```

### Issue: NumPy compatibility error
```bash
# Already fixed in requirements.txt
pip install "numpy<2.0"
```

### Issue: Can't find venv312
Make sure you're in the correct directory:
```bash
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System
dir venv312  # Should show the directory
```

---

## Next Steps

Once everything is installed:

1. **Run Quick Start**: `python examples/quick_start.py`
2. **Train Models**: `python scripts/train_models.py`
3. **Start API**: `python -m src.api.app`
4. **Explore Code**: Read the source files in `src/`
5. **Experiment**: Modify parameters and try different datasets

---

## Summary

```bash
# Complete setup (copy-paste these commands)
cd C:\Users\hugot\Documents\GitHub\ML-Projects\Recommendation-System
py -3.12 -m venv venv312
venv312\Scripts\activate
pip install -r requirements.txt
pip install -e .
python examples/quick_start.py
```

**That's it!** 🎉

Your recommendation system is now fully functional with:
- ✅ All 6 ML algorithms
- ✅ Data processing pipeline
- ✅ Evaluation framework
- ✅ REST API
- ✅ Web interface

Enjoy building recommendations! 🚀
