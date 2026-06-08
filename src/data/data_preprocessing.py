import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# Logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)       # FIX: was missing — WordNetLemmatizer needs this


def preprocess_comment(comment):
    """Apply preprocessing transformations to a single comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Retain sentiment-critical negation words
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        logger.error('Error preprocessing comment: %s', e)
        return comment


def normalize_text(df):
    """Apply preprocess_comment to every row in the comment column."""
    try:
        df['comment'] = df['comment'].apply(preprocess_comment)  # FIX: was preprocess_text (undefined)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save processed splits to data/interim/."""
    try:
        interim_path = os.path.join(data_path, 'interim')
        os.makedirs(interim_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_path, 'train_processed.csv'), index=False)
        test_data.to_csv(os.path.join(interim_path, 'test_processed.csv'), index=False)

        logger.debug('Processed data saved to %s — Train: %d rows | Test: %d rows',
                     interim_path, len(train_data), len(test_data))
    except Exception as e:
        logger.error('Error saving processed data: %s', e)
        raise


def main():
    try:
        logger.debug('Starting data preprocessing...')

        # Resolve project root relative to this file (src/data/data_preprocessing.py)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        # FIX: read from data/interim/ not data/raw/ — ingestion saves splits there
        interim_path = os.path.join(project_root, 'data', 'interim')
        train_data = pd.read_csv(os.path.join(interim_path, 'train.csv'))
        test_data  = pd.read_csv(os.path.join(interim_path, 'test.csv'))
        logger.debug('Loaded train (%d) and test (%d) from data/interim/', len(train_data), len(test_data))

        train_processed = normalize_text(train_data)
        test_processed  = normalize_text(test_data)

        save_data(train_processed, test_processed, data_path=os.path.join(project_root, 'data'))

    except Exception as e:
        logger.error('Data preprocessing failed: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()