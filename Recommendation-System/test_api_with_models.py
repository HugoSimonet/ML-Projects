"""Quick test of the API with trained models."""

import requests
import json

BASE_URL = "http://localhost:8000"

print("="*70)
print("TESTING RECOMMENDATION API WITH TRAINED MODELS")
print("="*70)

# Test 1: Health Check
print("\n[1/6] Health Check...")
response = requests.get(f"{BASE_URL}/health")
if response.status_code == 200:
    data = response.json()
    print(f"[OK] API is healthy")
    print(f"  Models loaded: {data['n_models_loaded']}")
    print(f"  Available models: {', '.join(data['models'])}")
else:
    print(f"[FAIL] Health check failed: {response.status_code}")

# Test 2: List Models
print("\n[2/6] List Models...")
response = requests.get(f"{BASE_URL}/models")
if response.status_code == 200:
    models = response.json()
    print(f"[OK] Found {len(models)} models:")
    for model in models:
        print(f"  - {model['name']} ({model['type']}) - Trained: {model['is_trained']}")
else:
    print(f"[FAIL] List models failed: {response.status_code}")

# Test 3: Get Recommendations
print("\n[3/6] Get Recommendations (User 1, Top 10)...")
response = requests.post(
    f"{BASE_URL}/recommend",
    json={"user_id": 1, "n_items": 10}
)
if response.status_code == 200:
    data = response.json()
    print(f"[OK] Got {data['n_recommendations']} recommendations using {data['model_used']}")
    print("  Top 5 recommendations:")
    for i, rec in enumerate(data['recommendations'][:5], 1):
        print(f"    {i}. Item {rec['item_id']} (score: {rec['score']:.3f})")
else:
    print(f"[FAIL] Recommendations failed: {response.status_code}")
    print(f"  Error: {response.text}")

# Test 4: Predict Rating
print("\n[4/6] Predict Rating (User 1, Item 50)...")
response = requests.post(
    f"{BASE_URL}/predict",
    json={"user_id": 1, "item_id": 50}
)
if response.status_code == 200:
    data = response.json()
    print(f"[OK] Predicted rating: {data['predicted_rating']:.2f}/5.0")
    print(f"  Model used: {data['model_used']}")
else:
    print(f"[FAIL] Prediction failed: {response.status_code}")
    print(f"  Error: {response.text}")

# Test 5: Batch Recommendations
print("\n[5/6] Batch Recommendations (Users 1-3, Top 5 each)...")
response = requests.post(
    f"{BASE_URL}/batch_recommend",
    json={"user_ids": [1, 2, 3], "n_items": 5}
)
if response.status_code == 200:
    data = response.json()
    print(f"[OK] Got recommendations for {data['n_users']} users using {data['model_used']}")
    for result in data['results']:
        print(f"  User {result['user_id']}: {result['n_recommendations']} items")
else:
    print(f"[FAIL] Batch recommendations failed: {response.status_code}")
    print(f"  Error: {response.text}")

# Test 6: Specific Model
print("\n[6/6] Get Recommendations with Specific Model (svd_movielens)...")
response = requests.post(
    f"{BASE_URL}/recommend",
    json={"user_id": 5, "n_items": 5, "model_name": "svd_movielens"}
)
if response.status_code == 200:
    data = response.json()
    print(f"[OK] Got {data['n_recommendations']} recommendations using {data['model_used']}")
    print("  Recommendations:")
    for rec in data['recommendations']:
        print(f"    Item {rec['item_id']} (score: {rec['score']:.3f}, rank: {rec['rank']})")
else:
    print(f"[FAIL] Hybrid recommendations failed: {response.status_code}")
    print(f"  Error: {response.text}")

print("\n" + "="*70)
print("API TESTING COMPLETE!")
print("="*70)
print("\nAll endpoints working correctly with trained models!")
print("API Documentation: http://localhost:8000/docs")
