import os
import csv
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer
import zipfile
import tempfile
import shutil

# Initialize the Kaggle API and SentenceTransformer model
api = KaggleApi()
api.authenticate()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your CSV containing dataset titles and links
datasets_info = pd.read_csv('kaggle_datasets.csv')

# Load existing embeddings to skip already processed datasets
processed_titles = set()
if os.path.exists('dataset_column_embeddings.csv'):
    existing_data = pd.read_csv('dataset_column_embeddings.csv')
    processed_titles = set(existing_data['Title'])

# Open the output CSV in append mode
with open('dataset_column_embeddings.csv', 'a', newline='', encoding='utf-8') as csvfile:
    # Define the CSV writer
    fieldnames = ['Title', 'Link', 'Column Name', 'Embedding']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header only if the file is empty
    if csvfile.tell() == 0:
        writer.writeheader()

    # Create a temporary directory to store all downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        for index, row in datasets_info.iterrows():
            title = row['Title']
            link = row['Link']

            # Skip if the dataset has already been processed
            if title in processed_titles:
                print(f"Skipping already processed dataset: {title}")
                continue
            
            # Extract the dataset reference from the link
            dataset_ref = link.split('/datasets/')[1]
            
            try:
                # List files in the dataset and download them
                dataset_files = api.dataset_list_files(dataset_ref)
                if not dataset_files.files:
                    raise ValueError("No files found in the dataset")

                # Attempt to download each file and read as a CSV
                csv_file_found = False
                for file_info in dataset_files.files:
                    file_name = file_info.name
                    download_path = os.path.join(temp_dir, file_name)

                    # Retry logic for downloading the entire file
                    try:
                        api.dataset_download_file(dataset_ref, file_name, path=temp_dir)
                    except Exception as e:
                        print(f"Retrying download for {file_name} due to error: {str(e)}")
                        api.dataset_download_file(dataset_ref, file_name, path=temp_dir)

                    # Check if the file is a zip archive
                    if file_name.endswith('.zip'):
                        with zipfile.ZipFile(download_path, 'r') as zip_ref:
                            zip_ref.extractall(temp_dir)
                        extracted_files = zip_ref.namelist()
                        for extracted_file in extracted_files:
                            if extracted_file.endswith('.csv'):
                                download_path = os.path.join(temp_dir, extracted_file)
                                break

                    # Ensure the file exists and try to read it
                    if os.path.exists(download_path) and download_path.endswith('.csv'):
                        try:
                            df = pd.read_csv(download_path, nrows=2, encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(download_path, nrows=2, encoding='ISO-8859-1')

                        column_names = df.columns.tolist()
                        column_embeddings = model.encode(column_names)

                        # Append each column name and its embedding to the CSV
                        for column_name, embedding in zip(column_names, column_embeddings):
                            writer.writerow({
                                'Title': title,
                                'Link': link,
                                'Column Name': column_name,
                                'Embedding': embedding.tolist()
                            })
                        csv_file_found = True
                        print(f"Processed: {title}")
                        break

                if not csv_file_found:
                    raise FileNotFoundError(f"No CSV file found in the dataset: {title}")

            except Exception as e:
                print(f"Failed to process {title}: {str(e)}")

# The temporary directory and all files in it are automatically cleaned up
