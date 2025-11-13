# Recommendation System

A comprehensive, production-ready recommendation system supporting multiple algorithms and domains (movies, music, e-commerce). Built with Python, featuring collaborative filtering, content-based filtering, matrix factorization, deep learning, and hybrid approaches.

## Features

### Multiple Recommendation Algorithms
- **Collaborative Filtering** (User-based & Item-based)
- **Content-Based Filtering** (TF-IDF)
- **Matrix Factorization** (SVD & ALS)
- **Deep Learning** (Neural Collaborative Filtering with PyTorch)
- **Hybrid System** (Weighted & Rank-based fusion)

### Multi-Domain Support
- **Movies**: MovieLens (100K, 1M, 10M)
- **Music**: Last.fm dataset
- **E-commerce**: Amazon reviews, synthetic data
- Domain-agnostic architecture for easy extension

### Comprehensive Evaluation
- Rating prediction metrics (RMSE, MAE)
- Ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K)
- Diversity, coverage, and novelty metrics
- Cross-validation support
- Visualization and comparison tools

### Production Features
- FastAPI REST API
- Docker & Docker Compose support
- Web demo interface
- Batch and real-time recommendations
- Model persistence and loading
- Comprehensive logging
- Type hints and docstrings
- Unit tests

## Project Structure

```
Recommendation-System/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data directory
│   ├── raw/              # Raw datasets
│   ├── processed/        # Preprocessed data
│   └── external/         # External data sources
├── models/               # Saved models
├── src/                  # Source code
│   ├── data/            # Data loading and preprocessing
│   │   ├── dataset_loader.py
│   │   └── preprocessor.py
│   ├── models/          # Recommendation algorithms
│   │   ├── base.py
│   │   ├── collaborative_filtering.py
│   │   ├── content_based.py
│   │   ├── matrix_factorization.py
│   │   ├── deep_learning.py
│   │   └── hybrid.py
│   ├── evaluation/      # Evaluation metrics
│   │   └── metrics.py
│   ├── api/            # FastAPI application
│   │   └── app.py
│   ├── visualization/  # Plotting utilities
│   │   └── plots.py
│   └── utils/          # Utility functions
│       ├── config_loader.py
│       └── logger.py
├── scripts/            # Training and utility scripts
│   └── train_models.py
├── tests/             # Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_evaluation.py
├── web/               # Web demo interface
│   └── index.html
├── notebooks/         # Jupyter notebooks
├── docs/             # Documentation
├── Dockerfile        # Docker configuration
├── docker-compose.yml
├── requirements.txt  # Python dependencies
├── setup.py         # Package setup
└── README.md        # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) Docker & Docker Compose

### Local Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Recommendation-System.git
cd Recommendation-System
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install package**
```bash
pip install -e .
```

### Docker Installation

```bash
docker-compose up -d
```

This will start:
- API server on http://localhost:8000
- Web interface on http://localhost:8080

## Quick Start

### 1. Train Models

```bash
# Train all models on MovieLens 100K
python scripts/train_models.py --dataset movielens --version 100k --models all --save --visualize

# Train specific models
python scripts/train_models.py --dataset movielens --models svd als ncf --save

# Use synthetic data for testing
python scripts/train_models.py --dataset synthetic --models all --save
```

### 2. Using the API

**Start the API server:**
```bash
python -m src.api.app
```

**Make requests:**
```python
import requests

# Get recommendations
response = requests.post('http://localhost:8000/recommend', json={
    'user_id': 1,
    'n_items': 10,
    'exclude_seen': True
})
recommendations = response.json()

# Predict rating
response = requests.post('http://localhost:8000/predict', json={
    'user_id': 1,
    'item_id': 50
})
prediction = response.json()
```

**API Documentation:**
Visit http://localhost:8000/docs for interactive API documentation.

### 3. Using the Web Interface

1. Start the API server
2. Open `web/index.html` in your browser
3. Enter user ID and get recommendations

### 4. Python Usage

```python
from src.data.dataset_loader import DatasetLoader
from src.data.preprocessor import DataPreprocessor
from src.models.hybrid import HybridRecommender
from src.models.collaborative_filtering import UserBasedCF
from src.models.matrix_factorization import SVDRecommender
from src.evaluation.metrics import Evaluator

# Load data
loader = DatasetLoader('movielens', '100k')
data = loader.load()
ratings_df = data['ratings']

# Preprocess
preprocessor = DataPreprocessor()
processed = preprocessor.preprocess_pipeline(ratings_df)
train_df = processed['train']
test_df = processed['test']

# Train models
user_cf = UserBasedCF()
user_cf.fit(train_df)

svd = SVDRecommender()
svd.fit(train_df)

# Create hybrid
hybrid = HybridRecommender(
    models={'user_cf': user_cf, 'svd': svd},
    weights={'user_cf': 0.5, 'svd': 0.5}
)

# Get recommendations
recommendations = hybrid.recommend(user_id=1, n_items=10)
print(recommendations)

# Evaluate
evaluator = Evaluator()
results = evaluator.evaluate_all(hybrid, test_df, set(train_df['item_id'].unique()))
print(results)
```

## Configuration

Edit `config/config.yaml` to customize:
- Dataset URLs and paths
- Model hyperparameters
- Training settings
- Evaluation metrics
- API configuration

## API Endpoints

### Core Endpoints

- `POST /recommend` - Get personalized recommendations
- `POST /predict` - Predict rating for user-item pair
- `GET /recommend/{user_id}` - Simple recommendation endpoint
- `POST /batch_recommend` - Batch recommendations for multiple users

### Utility Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /models` - List loaded models
- `POST /load_model` - Load model from disk

## Evaluation Metrics

### Rating Prediction
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)

### Ranking
- **Precision@K** - Fraction of recommended items that are relevant
- **Recall@K** - Fraction of relevant items that are recommended
- **F1@K** - Harmonic mean of Precision and Recall
- **NDCG@K** (Normalized Discounted Cumulative Gain) - Position-aware relevance
- **MAP@K** (Mean Average Precision) - Mean of average precisions

### Diversity & Coverage
- **Diversity** - Variety of recommendations across users
- **Coverage** - Fraction of items that can be recommended
- **Novelty** - How surprising the recommendations are

## Datasets

### MovieLens
Automatically downloaded when running training scripts.

**Versions:**
- 100K: 100,000 ratings from 943 users on 1,682 movies
- 1M: 1 million ratings from 6,040 users on 3,706 movies
- 10M: 10 million ratings from 71,567 users on 10,681 movies

### Last.fm
Download manually from: http://ocelma.net/MusicRecommendationDataset/lastfm-360K.html

Place in `data/raw/lastfm-360K/`

### Amazon Reviews
Download from: https://nijianmo.github.io/amazon/index.html

Place in `data/raw/amazon/{category}/`

### Synthetic Data
Generated automatically for testing purposes.

## Model Details

### Collaborative Filtering
Uses user-user or item-item similarity with K-nearest neighbors.
- **Advantages**: Simple, interpretable, works well with explicit feedback
- **Disadvantages**: Cold start problem, sparsity issues

### Content-Based Filtering
Uses TF-IDF to create item profiles and find similar items.
- **Advantages**: No cold start for items with features, personalized
- **Disadvantages**: Limited serendipity, requires item features

### Matrix Factorization (SVD/ALS)
Decomposes rating matrix into latent factors.
- **Advantages**: Handles sparsity, scalable, good performance
- **Disadvantages**: Harder to interpret, cold start for new users/items

### Neural Collaborative Filtering
Deep learning approach using embeddings and neural networks.
- **Advantages**: Can learn complex patterns, state-of-the-art performance
- **Disadvantages**: Requires more data, computationally expensive

### Hybrid System
Combines multiple algorithms using weighted or rank-based fusion.
- **Advantages**: Best of all worlds, robust, better coverage
- **Disadvantages**: More complex, requires tuning weights

## Testing

Run unit tests:
```bash
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Docker Usage

### Build and run
```bash
docker-compose up -d
```

### View logs
```bash
docker-compose logs -f api
```

### Stop services
```bash
docker-compose down
```

### Build individual container
```bash
docker build -t recsys .
docker run -p 8000:8000 -v $(pwd)/models:/app/models recsys
```

## Performance Tips

1. **Use ALS for large datasets** - Optimized for implicit feedback
2. **Enable GPU for NCF** - Significantly faster training
3. **Precompute similarities** - For collaborative filtering
4. **Use batch recommendations** - More efficient than individual requests
5. **Cache predictions** - For frequently requested items
6. **Sample data for tuning** - Use subset for hyperparameter optimization

## Troubleshooting

### Import errors
```bash
pip install -e .
```

### CUDA/GPU issues
Set environment variable:
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU only
```

### Memory errors
- Reduce batch size for NCF
- Use smaller dataset version
- Reduce number of factors in matrix factorization

### Slow training
- Use fewer epochs
- Reduce K in collaborative filtering
- Use ALS instead of SVD for large datasets

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## Citation

If you use this project in your research, please cite:

```bibtex
@software{recommendation_system,
  author = {Your Name},
  title = {Comprehensive Recommendation System},
  year = {2024},
  url = {https://github.com/yourusername/Recommendation-System}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MovieLens dataset: GroupLens Research
- Surprise library for collaborative filtering
- Implicit library for ALS implementation
- FastAPI for the excellent web framework

## Contact

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/Recommendation-System

## References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems.
2. He, X., et al. (2017). Neural collaborative filtering.
3. Sarwar, B., et al. (2001). Item-based collaborative filtering recommendation algorithms.
4. Pazzani, M. J., & Billsus, D. (2007). Content-based recommendation systems.

---

**Happy Recommending!**
