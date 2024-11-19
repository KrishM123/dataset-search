import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_dataset_info():
    """Load both raw dataset info and embeddings from JSONL files"""
    raw_datasets = {}
    embeddings = {}
    
    # Load raw dataset information
    with open('raw_datasets.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_datasets[data['dataset_id']] = data
    
    # Load embeddings
    with open('embeddings.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            embeddings[data['dataset_id']] = data['embeddings']
    
    return raw_datasets, embeddings

def extract_required_columns(prompt: str, model: SentenceTransformer, similarity_threshold: float = 0.5) -> list:
    """
    Extract the required columns from a natural language ML task description using embeddings.
    """
    print("\nAnalyzing prompt for required column types...")
    
    common_ml_columns = {
        'target': ['target', 'label', 'output', 'result', 'prediction'],
        'category': ['category', 'class', 'type', 'group', 'classification'],
        'numeric': ['value', 'amount', 'price', 'score', 'rating', 'number'],
        'feature': ['feature', 'input', 'attribute', 'characteristic', 'property'],
        'temporal': ['date', 'time', 'year', 'month', 'timestamp'],
        'identifier': ['id', 'user', 'item', 'product', 'customer']
    }
    
    # Generate embeddings for the prompt
    prompt_embedding = model.encode(prompt, convert_to_tensor=True).cpu().numpy()
    
    # Generate embeddings for all column types and their variations
    column_type_embeddings = {}
    for column_type, variations in common_ml_columns.items():
        # Combine variations into a single description
        description = f"{column_type}: {', '.join(variations)}"
        embedding = model.encode(description, convert_to_tensor=True).cpu().numpy()
        column_type_embeddings[column_type] = embedding
    
    # Calculate similarities
    required_columns = []
    for column_type, embedding in column_type_embeddings.items():
        similarity = float(cosine_similarity(
            prompt_embedding.reshape(1, -1),
            embedding.reshape(1, -1)
        )[0][0])
        
        if similarity >= similarity_threshold:
            required_columns.append({
                'type': column_type,
                'similarity': similarity
            })
    
    # Sort by similarity
    required_columns.sort(key=lambda x: x['similarity'], reverse=True)
    
    if required_columns:
        print("\nIdentified required column types:")
        for col in required_columns:
            print(f"  - {col['type']} (similarity: {col['similarity']:.3f})")
    else:
        print("No specific column types identified from the prompt")
    
    return required_columns

def find_matching_dataset_columns(prompt: str, required_columns: list, model: SentenceTransformer, similarity_threshold: float = 0.5):
    """
    Find all matching dataset columns above a similarity threshold for a given ML task description.
    Now considers both direct prompt similarity and required column type similarity.
    """
    print(f"\nSearching for columns matching: '{prompt}'")
    
    # Load dataset information
    print("Loading dataset information...")
    raw_datasets, embeddings = load_dataset_info()
    print(f"Loaded information for {len(raw_datasets)} datasets")
    
    print("Generating embedding for input prompt...")
    prompt_embedding = model.encode(prompt, convert_to_tensor=True).cpu().numpy()
    
    results = {
        'datasets': []
    }
    
    print("Computing similarities with datasets...")
    for dataset_id in tqdm(raw_datasets.keys(), desc="Processing datasets"):
        raw_data = raw_datasets[dataset_id]
        dataset_embeddings = embeddings[dataset_id]
        
        # Calculate title similarity
        title_similarity = float(cosine_similarity(
            prompt_embedding.reshape(1, -1),
            np.array(dataset_embeddings['title']).reshape(1, -1)
        )[0][0])
        
        # Calculate column similarities
        matching_columns = []
        for column_data in dataset_embeddings['columns']:
            column_embedding = np.array(column_data['embedding'])
            
            # Calculate direct similarity with prompt
            prompt_similarity = float(cosine_similarity(
                prompt_embedding.reshape(1, -1),
                column_embedding.reshape(1, -1)
            )[0][0])
            
            # Calculate similarities with required column types
            type_similarities = []
            for req_col in required_columns:
                type_sim = float(cosine_similarity(
                    model.encode(req_col['type'], convert_to_tensor=True).cpu().numpy().reshape(1, -1),
                    column_embedding.reshape(1, -1)
                )[0][0])
                type_similarities.append(type_sim)
            
            # Use maximum similarity between prompt and column types
            max_similarity = max([prompt_similarity] + type_similarities)
            
            if max_similarity >= similarity_threshold:
                matching_columns.append({
                    'name': column_data['name'],
                    'similarity': max_similarity,
                    'prompt_similarity': prompt_similarity,
                    'type_similarity': max(type_similarities) if type_similarities else 0
                })
        
        # If we have matching columns or high title similarity, include the dataset
        if matching_columns or title_similarity >= similarity_threshold:
            dataset_info = {
                'dataset_id': dataset_id,
                'dataset_title': raw_data['title'],
                'dataset_link': raw_data['link'],
                'title_similarity': title_similarity,
                'columns': sorted(matching_columns, key=lambda x: x['similarity'], reverse=True)
            }
            results['datasets'].append(dataset_info)
    
    # Sort datasets by best matching column or title similarity
    results['datasets'].sort(key=lambda x: max(
        [col['similarity'] for col in x['columns']] + [x['title_similarity']]
    ), reverse=True)
    
    print(f"\nFound {len(results['datasets'])} relevant datasets:")
    for dataset in results['datasets']:
        print(f"\nDataset: {dataset['dataset_title']}")
        print(f"Link: {dataset['dataset_link']}")
        print(f"Title Similarity: {dataset['title_similarity']:.3f}")
        if dataset['columns']:
            print("Relevant columns:")
            for col in dataset['columns']:
                print(f"  - {col['name']}")
                print(f"    Prompt similarity: {col['prompt_similarity']:.3f}")
                print(f"    Type similarity: {col['type_similarity']:.3f}")
                print(f"    Overall similarity: {col['similarity']:.3f}")
    
    return results

def search_datasets_for_ml_task(prompt: str, similarity_threshold: float = 0.5) -> dict:
    """
    Main function to search for appropriate datasets given an ML task description.
    """
    print("\n=== Starting Dataset Search for ML Task ===")
    print(f"Task description: '{prompt}'")
    
    # Initialize the model once and pass it to other functions
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract required columns from the prompt using embeddings
    required_columns = extract_required_columns(prompt, model, similarity_threshold)
    
    # Find matching columns using semantic search
    matches = find_matching_dataset_columns(prompt, required_columns, model, similarity_threshold)
    
    # Add the required columns to the results
    matches['required_columns'] = required_columns
    
    print("\n=== Search Complete ===")
    return matches
