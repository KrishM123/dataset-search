# Kaggle Dataset Search

A tool to semantically search Kaggle datasets based on column names and dataset titles.

## Setup

1. **Create and activate a conda environment:**
   ```bash
   conda create -n kaggle-search python=3.11
   conda activate kaggle-search
   ```

2. **Install requirements:**
   Make sure you have a `requirements.txt` file in your project directory. If you don't have it, create one with the following content:
   ```plaintext
   kaggle
   pandas
   sentence-transformers
   scikit-learn
   beautifulsoup4
   tqdm
   ```
   Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API credentials:**
   - Go to [kaggle.com](https://www.kaggle.com) → Account → Create New API Token.
   - Download `kaggle.json`.
   - Place it in `~/.kaggle/kaggle.json` (Windows: `C:\Users\<username>\.kaggle\kaggle.json`).

## Data Collection and Processing

1. **Collect dataset information:**
   Run the following command to gather dataset titles and links:
   ```bash
   python directory_parse.py
   ```
   This will create `kaggle_datasets.csv` containing dataset titles and links.

2. **Process the datasets and generate embeddings:**
   Run the following command to process the datasets and create embeddings:
   ```bash
   python column_parse.py
   ```
   This will create:
   - `raw_datasets.jsonl`: Contains raw dataset information.
   - `embeddings.jsonl`: Contains embeddings for titles and columns.

## Searching Datasets

1. **Create a `main.py` file** with your search query:
   ```python
   from search import search_datasets_for_ml_task

   # Example queries:
   query = "Find a dataset for predicting house prices based on features like size and location"
   # or
   query = "Looking for user-item interactions data for building a recommendation system"

   results = search_datasets_for_ml_task(query, similarity_threshold=0.5)
   ```

2. **Run the search:**
   Execute the following command:
   ```bash
   python main.py
   ```

## Notes

- The initial data collection and processing may take several hours depending on the number of datasets.
- Make sure you have enough disk space for downloading datasets.
- The search uses semantic similarity, so it can understand natural language queries.
