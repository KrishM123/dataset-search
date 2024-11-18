import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def find_matching_dataset_columns(prompt: str, embedding_csv_path: str = "dataset_column_embeddings.csv", similarity_threshold: float = 0.5):
    """
    Find all matching dataset columns above a similarity threshold for a given ML task description.
    
    Args:
        prompt (str): Natural language description of the ML task
        embedding_csv_path (str): Path to the CSV file containing pre-computed column embeddings
        similarity_threshold (float): Minimum similarity score to consider a column match (0-1)
    """
    print(f"\nSearching for columns matching: '{prompt}'")
    
    try:
        print(f"Loading embeddings from {embedding_csv_path}...")
        df_embeddings = pd.read_csv(embedding_csv_path)
        print(f"Loaded {len(df_embeddings)} column embeddings from {len(df_embeddings['Title'].unique())} datasets")
    except FileNotFoundError:
        raise FileNotFoundError(f"Embeddings file not found at {embedding_csv_path}")
    
    print("Initializing sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embedding for input prompt...")
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    
    print("Computing similarities with dataset columns...")
    column_embeddings = []
    
    for emb in tqdm(df_embeddings['Embedding'].values, 
                   desc="Processing embeddings", 
                   unit="column"):
        column_embeddings.append(eval(emb))
    
    column_embeddings = np.array(column_embeddings)
    
    print("Calculating cosine similarities...")
    similarities = cosine_similarity(
        prompt_embedding.cpu().numpy().reshape(1, -1), 
        column_embeddings
    )[0]
    
    # Create a DataFrame with similarities for easier processing
    df_embeddings['similarity'] = similarities
    
    # Filter columns above threshold and sort by similarity
    relevant_columns = df_embeddings[df_embeddings['similarity'] >= similarity_threshold]
    
    # Group by dataset
    dataset_groups = relevant_columns.groupby('Title')
    
    results = {
        'datasets': []
    }
    
    print(f"\nFound {len(dataset_groups)} datasets with relevant columns:")
    
    for dataset_title, group in dataset_groups:
        # Sort columns within each dataset by similarity
        group = group.sort_values('similarity', ascending=False)
        
        dataset_info = {
            'dataset_title': dataset_title,
            'dataset_link': group.iloc[0]['Link'],  # All rows have same link
            'columns': [
                {
                    'name': row['Column Name'],
                    'similarity': float(row['similarity'])
                }
                for _, row in group.iterrows()
            ]
        }
        
        results['datasets'].append(dataset_info)
        
        # Print detailed results
        print(f"\nDataset: {dataset_title}")
        print(f"Link: {dataset_info['dataset_link']}")
        print("Relevant columns:")
        for col in dataset_info['columns']:
            print(f"  - {col['name']} (similarity: {col['similarity']:.3f})")
    
    return results

def extract_required_columns(prompt: str) -> list:
    """
    Extract the required columns from a natural language ML task description.
    """
    print("\nAnalyzing prompt for required column types...")
    
    common_ml_keywords = {
        'predict': ['target', 'label'],
        'classify': ['category', 'class', 'label'],
        'regression': ['value', 'target'],
        'cluster': ['features'],
        'recommend': ['user', 'item', 'rating']
    }
    
    extracted_columns = []
    prompt_lower = prompt.lower()
    
    for keyword, columns in common_ml_keywords.items():
        if keyword in prompt_lower:
            extracted_columns.extend(columns)
    
    if extracted_columns:
        print(f"Required column types identified: {', '.join(extracted_columns)}")
    else:
        print("No specific column types identified from the prompt")
    
    return list(set(extracted_columns))

def search_datasets_for_ml_task(prompt: str, similarity_threshold: float = 0.5) -> dict:
    """
    Main function to search for appropriate datasets given an ML task description.
    
    Args:
        prompt (str): Natural language description of the ML task
        similarity_threshold (float): Minimum similarity score to consider a column match (0-1)
    """
    print("\n=== Starting Dataset Search for ML Task ===")
    print(f"Task description: '{prompt}'")
    
    # Extract required columns from the prompt
    required_columns = extract_required_columns(prompt)
    
    # Find matching columns using semantic search
    matches = find_matching_dataset_columns(prompt, similarity_threshold=similarity_threshold)
    
    # Add the required columns to the results
    matches['required_columns'] = required_columns
    
    print("\n=== Search Complete ===")
    return matches
