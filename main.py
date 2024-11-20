from search import search_datasets_for_ml_task

# Example queries:
query = "Find a dataset for predicting house prices based on features like size and location"
# or
query = "Looking for user-item interactions data for building a recommendation system"

results = search_datasets_for_ml_task(query, similarity_threshold=0.5)