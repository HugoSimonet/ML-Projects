"""Examples of sending POST requests to the Recommendation API."""

import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

print("=" * 70)
print("POST REQUEST EXAMPLES")
print("=" * 70)

# Example 1: Simple recommendation request
print("\n1. SIMPLE RECOMMENDATION REQUEST")
print("-" * 70)

response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "user_id": 1,
        "n_items": 5
    }
)

print(f"Status Code: {response.status_code}")
print(f"Response:")
print(json.dumps(response.json(), indent=2))

# Example 2: Recommendation with specific model
print("\n2. RECOMMENDATION WITH SPECIFIC MODEL (SVD - Best Performer)")
print("-" * 70)

response = requests.post(
    f"{BASE_URL}/recommend",
    json={
        "user_id": 5,
        "n_items": 10,
        "model_name": "svd_movielens"
    }
)

print(f"Status Code: {response.status_code}")
result = response.json()
print(f"Model used: {result['model_used']}")
print(f"Got {result['n_recommendations']} recommendations")
print(f"Top 3:")
for rec in result['recommendations'][:3]:
    print(f"  - Item {rec['item_id']}: score {rec['score']:.3f}")

# Example 3: Predict rating
print("\n3. PREDICT RATING FOR USER-ITEM PAIR")
print("-" * 70)

response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "user_id": 1,
        "item_id": 100
    }
)

print(f"Status Code: {response.status_code}")
result = response.json()
print(f"User {result['user_id']}, Item {result['item_id']}")
print(f"Predicted Rating: {result['predicted_rating']:.2f}/5.0")
print(f"Model used: {result['model_used']}")

# Example 4: Batch recommendations
print("\n4. BATCH RECOMMENDATIONS (Multiple Users)")
print("-" * 70)

response = requests.post(
    f"{BASE_URL}/batch_recommend",
    json={
        "user_ids": [1, 2, 3, 4, 5],
        "n_items": 5
    }
)

print(f"Status Code: {response.status_code}")
result = response.json()
print(f"Processed {result['n_users']} users using {result['model_used']}")
for user_result in result['results'][:3]:  # Show first 3
    print(f"  User {user_result['user_id']}: {user_result['n_recommendations']} items")

# Example 5: Error handling
print("\n5. ERROR HANDLING (Invalid User ID)")
print("-" * 70)

response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "user_id": -1,  # Invalid
        "item_id": 50
    }
)

print(f"Status Code: {response.status_code}")
if response.status_code != 200:
    print(f"Error: {response.json()}")
else:
    print(f"Response: {response.json()}")

print("\n" + "=" * 70)
print("TIP: For interactive testing, visit http://localhost:8000/docs")
print("=" * 70)
