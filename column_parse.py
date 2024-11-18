import os
import json
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer
import zipfile
import tempfile
import uuid

# Initialize the Kaggle API and SentenceTransformer model
api = KaggleApi()
api.authenticate()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your CSV containing dataset titles and links
datasets_info = pd.read_csv('kaggle_datasets.csv')

# Load existing dataset IDs to skip already processed datasets
processed_ids = set()
if os.path.exists('embeddings.jsonl'):
    with open('embeddings.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            processed_ids.add(data['dataset_id'])

# Open both output files in append mode
with open('raw_datasets.jsonl', 'a', encoding='utf-8') as raw_file, \
     open('embeddings.jsonl', 'a', encoding='utf-8') as embeddings_file:
    
    # Create a temporary directory to store all downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, row in datasets_info.iterrows():
            title = row['Title']
            link = row['Link']
            
            # Generate a unique ID for the dataset
            dataset_id = str(uuid.uuid4())
            
            # Skip if the dataset has already been processed
            if dataset_id in processed_ids:
                print(f"Skipping already processed dataset: {title}")
                continue
            
            # Extract the dataset reference from the link
            dataset_ref = link.split('/datasets/')[1]
            
            try:
                # Initialize the raw dataset record
                raw_record = {
                    "dataset_id": dataset_id,
                    "title": title,
                    "link": link,
                    "columns": []
                }
                
                # Initialize the embeddings record
                embeddings_record = {
                    "dataset_id": dataset_id,
                    "embeddings": {
                        "title": model.encode(title).tolist(),
                        "columns": []
                    }
                }
                
                # List files in the dataset and download them
                dataset_files = api.dataset_list_files(dataset_ref)
                if not dataset_files.files:
                    raise ValueError("No files found in the dataset")

                # Attempt to download each file and read as a CSV
                csv_file_found = False
                for file_info in dataset_files.files:
                    file_name = file_info.name
                    download_path = os.path.join(temp_dir, file_name)

                    try:
                        api.dataset_download_file(dataset_ref, file_name, path=temp_dir)
                    except Exception as e:
                        print(f"Retrying download for {file_name} due to error: {str(e)}")
                        api.dataset_download_file(dataset_ref, file_name, path=temp_dir)

                    # Handle zip files
                    if file_name.endswith('.zip'):
                        with zipfile.ZipFile(download_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        extracted_files = zip_ref.namelist()
                        for extracted_file in extracted_files:
                            if extracted_file.endswith('.csv'):
                                download_path = os.path.join(temp_dir, extracted_file)
                                break

                    # Process CSV files
                    if os.path.exists(download_path) and download_path.endswith('.csv'):
                        try:
                            df = pd.read_csv(download_path, nrows=2, encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(download_path, nrows=2, encoding='ISO-8859-1')

                        column_names = df.columns.tolist()
                        
                        # Process each column individually
                        successful_columns = []
                        for column_name in column_names:
                            try:
                                # Generate embedding for single column
                                column_embedding = model.encode(column_name).tolist()
                                
                                # Add successful column to raw record
                                successful_columns.append(column_name)
                                
                                # Add column embedding to embeddings record
                                embeddings_record["embeddings"]["columns"].append({
                                    "name": column_name,
                                    "embedding": column_embedding
                                })
                            except Exception as e:
                                print(f"Failed to process column '{column_name}' in dataset '{title}': {str(e)}")
                                continue
                        
                        # Update raw record with successful columns
                        raw_record["columns"] = successful_columns
                        
                        if successful_columns:  # If at least one column was processed
                            csv_file_found = True
                            print(f"Processed: {title} ({len(successful_columns)}/{len(column_names)} columns)")
                            break
                        else:
                            print(f"Warning: No columns could be processed in dataset: {title}")

                if not csv_file_found:
                    raise FileNotFoundError(f"No CSV file found in the dataset: {title}")

                # Write both records to their respective files
                raw_file.write(json.dumps(raw_record) + '\n')
                raw_file.flush()
                
                embeddings_file.write(json.dumps(embeddings_record) + '\n')
                embeddings_file.flush()

            except Exception as e:
                print(f"Failed to process {title}: {str(e)}")

# The temporary directory and all files in it are automatically cleaned up