import csv
import time
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Open the CSV file in append mode with utf-8 encoding
with open('kaggle_datasets.csv', 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Title', 'Link']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write the header only if the file is empty
    if csvfile.tell() == 0:
        writer.writeheader()

    page = 1

    while True:
        datasets = api.dataset_list(page=page)
        if not datasets:
            break
        
        for dataset in datasets:
            ref = dataset.ref
            title = dataset.title
            link = f"https://www.kaggle.com/datasets/{ref}"
            
            # Write each dataset to the CSV file
            writer.writerow({'Title': title, 'Link': link})
        
        print(f"Retrieved {len(datasets)} datasets from page {page}")
        page += 1
        time.sleep(1)  # To respect API rate limits

print("Dataset links saved to kaggle_datasets.csv")
