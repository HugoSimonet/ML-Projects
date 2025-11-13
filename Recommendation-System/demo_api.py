"""Simple demo of the Recommendation API."""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("RECOMMENDATION API DEMO")
print("=" * 70)

# 1. Check API health
print("\n1. Checking API Health...")
response = requests.get(f"{BASE_URL}/health")
data = response.json()
print(f"   Status: {data['status']}")
print(f"   Models loaded: {data['n_models_loaded']}")

# 2. Get recommendations for a user
print("\n2. Getting recommendations for User #1...")
response = requests.post(
    f"{BASE_URL}/recommend",
    json={"user_id": 1, "n_items": 5}
)
data = response.json()
print(f"   Using model: {data['model_used']}")
print(f"   Top 5 recommendations:")
for rec in data['recommendations']:
    print(f"      - Item {rec['item_id']}: score {rec['score']:.3f}")

# 3. Predict a specific rating
print("\n3. Predicting rating for User #1, Item #50...")
response = requests.post(
    f"{BASE_URL}/predict",
    json={"user_id": 1, "item_id": 50}
)
data = response.json()
print(f"   Predicted rating: {data['predicted_rating']:.2f}/5.0")
print(f"   Model used: {data['model_used']}")

# 4. Try the best model (SVD)
print("\n4. Using best model (SVD) for User #5...")
response = requests.post(
    f"{BASE_URL}/recommend",
    json={"user_id": 5, "n_items": 5, "model_name": "svd_movielens"}
)
data = response.json()
print(f"   Top 5 recommendations from SVD:")
for rec in data['recommendations']:
    print(f"      - Item {rec['item_id']}: score {rec['score']:.3f}")

# 5. Batch recommendations
print("\n5. Getting batch recommendations for Users 1, 2, 3...")
response = requests.post(
    f"{BASE_URL}/batch_recommend",
    json={"user_ids": [1, 2, 3], "n_items": 3}
)
data = response.json()
print(f"   Processed {data['n_users']} users")
for result in data['results']:
    print(f"   User {result['user_id']}: {result['n_recommendations']} items")

print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print("\nTo explore more, visit: http://localhost:8000/docs")
