import json
import pinecone
from typing import Dict, List
from sentence_transformers import SentenceTransformer
import os

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def load_datasets() -> Dict[str, Dict]:
    """Load and pair the dataset metadata with embeddings"""
    metadata = {}
    embeddings = {}
    source = "kaggle"
    
    # Load raw dataset metadata
    with open('./raw_datasets.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            metadata[data['dataset_id']] = {
                "source" : source,
                'title': data['title'],
                'link': data['link'],
                'columns': data['columns']
            }
    
    # Load embeddings
    with open('./embeddings.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            embeddings[data['dataset_id']] = data['embeddings']['title']
    
    return metadata, embeddings

from pinecone import Pinecone

def upload_to_pinecone(
    index_name: str,
    metadata: Dict[str, Dict],
    embeddings: Dict[str, List[float]],
    batch_size: int = 100
):
    """Upload dataset embeddings and metadata to Pinecone"""
    # Initialize Pinecone
    pc = Pinecone(
        api_key=PINECONE_API_KEY, 
        environment="us-east-1"    # Replace with your environment
    )
    
    
    # Get index
    index = pc.Index(index_name)
    
    # Prepare vectors for upload
    vectors = []
    for dataset_id in metadata.keys():
        if dataset_id in embeddings:
            vectors.append((
                dataset_id,
                embeddings[dataset_id],
                metadata[dataset_id]
            ))

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=[(id, emb, meta) for id, emb, meta in batch])
        print(f"Uploaded {i + len(batch)} vectors")

def upload_ds():
        metadata, embeddings = load_datasets()
        upload_to_pinecone(
            index_name="datasets",  # Choose your index name
            metadata=metadata,
            embeddings=embeddings
        )

def query_index(
    query_text: str, 
    similarity_threshold: float = 0.5,
    top_k: int = 100  # Increased to get more candidates before filtering
) -> List[Dict]:
    """
    Query the Pinecone index and return matches above similarity threshold
    
    Args:
        query_embedding: Vector embedding of the query
        similarity_threshold: Minimum similarity score (0-1) to include in results
        top_k: Maximum number of results to consider
    
    Returns:
        List of dictionaries containing matches with scores and metadata,
        filtered by similarity threshold and sorted by score
    """
    # Initialize Pinecone
    pc = Pinecone(
        api_key=PINECONE_API_KEY,
        environment="us-east-1"
    )
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query_text).tolist()
    # Get index
    index = pc.Index("datasets")
    
    # Query the index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Format and filter results
    matches = []
    for match in results['matches']:
        if match.score >= similarity_threshold:  # Only include matches above threshold
            matches.append({
                'dataset_id': match.id,
                'score': match.score,
                'title': match.metadata['title'],
                'link': match.metadata['link'],
                'columns': match.metadata['columns']
            })

    if len(matches) < 5:
        matches = []  # Clear existing matches
        for match in results['matches'][:5]:  # Take top 5 matches
            matches.append({
                'dataset_id': match.id,
                'score': match.score,
                'title': match.metadata['title'],
                'link': match.metadata['link'],
                'columns': match.metadata['columns']
            })

    
    return matches


#response = query_index("Find datasets about AirBnB listings") 
#print(response)

#upload_ds()
    
