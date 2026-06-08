import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging
import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))

# Logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from local CSV file(s) in data/raw/."""
    try:
        # Pick up any CSV in the folder
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError(f'No CSV files found in {data_path}')
        
        csv_path = os.path.join(data_path, csv_files[0])
        logger.debug('Loading dataset: %s', csv_path)
        
        df = pd.read_csv(csv_path)
        logger.debug('Data loaded. Shape: %s | Columns: %s', df.shape, df.columns.tolist())
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename dataset columns to standard names: 'comment' and 'sentiment'.
    Handles common column name variations from different datasets.
    """
    try:
        # Map of possible column names → standard names
        comment_aliases  = ['comment', 'Comment', 'clean_comment', 'text', 'Text',
                            'comment_text', 'Comment_Text', 'body','CommentText', 'content']
        sentiment_aliases = ['sentiment', 'Sentiment', 'label', 'Label',
                             'category', 'Category', 'class', 'polarity']

        col_map = {}
        for col in df.columns:
            if col in comment_aliases:
                col_map[col] = 'comment'
            elif col in sentiment_aliases:
                col_map[col] = 'sentiment'

        df = df.rename(columns=col_map)

        # Verify both required columns now exist
        if 'comment' not in df.columns:
            raise KeyError(
                f"Could not find a comment column. Available columns: {df.columns.tolist()}\n"
                "Add its name to the 'comment_aliases' list in standardize_columns()."
            )
        if 'sentiment' not in df.columns:
            raise KeyError(
                f"Could not find a sentiment/label column. Available columns: {df.columns.tolist()}\n"
                "Add its name to the 'sentiment_aliases' list in standardize_columns()."
            )

        # Keep only the two columns we need
        df = df[['comment', 'sentiment']]
        logger.debug('Columns standardized. Sample labels: %s', df['sentiment'].unique()[:5])
        return df

    except Exception as e:
        logger.error('Column standardization failed: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove missing values, duplicates, and blank comment rows."""
    try:
        before = len(df)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = df[df['comment'].str.strip() != '']
        after = len(df)
        logger.debug('Preprocessing done. Rows before: %d | after: %d | removed: %d',
                     before, after, before - after)
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save train and test splits to data/interim/ (NOT data/raw/)."""
    try:
        # ← KEY FIX: save to interim, not raw
        interim_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(interim_path, 'test.csv'), index=False)

        logger.debug('Train (%d rows) and Test (%d rows) saved to %s',
                     len(train_data), len(test_data), interim_path)
    except Exception as e:
        logger.error('Unexpected error while saving data: %s', e)
        raise

def main():
    try:
        # Resolve paths relative to this file's location (src/data/)
        src_data_dir  = os.path.dirname(os.path.abspath(__file__))   # .../src/data/
        project_root  = os.path.abspath(os.path.join(src_data_dir, '../../'))

        # Load params
        params = load_params(os.path.join(project_root, 'params.yaml'))
        test_size = params['data_ingestion']['test_size']

        # Load raw data from data/raw/
        raw_data_path = os.path.join(project_root, 'data', 'raw')
        df = load_data(raw_data_path)

        # Standardize column names → 'comment', 'sentiment'
        df = standardize_columns(df)

        # Clean the data
        df = preprocess_data(df)

        # Train / test split
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logger.debug('Split done. Train: %d | Test: %d', len(train_data), len(test_data))

        # Save to data/interim/
        save_data(train_data, test_data, data_path=os.path.join(project_root, 'data'))

    except Exception as e:
        logger.error('Data ingestion failed: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()