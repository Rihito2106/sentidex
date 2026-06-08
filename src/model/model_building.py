import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import mlflow
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
import dotenv

dotenv.load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

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


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF with ngrams to the data."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

        X_train = train_data['comment'].values
        y_train = train_data['sentiment'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Train shape: {X_train_tfidf.shape}")

        # Save the vectorizer to models/ folder
        models_dir = os.path.join(get_root_directory(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF applied and vectorizer saved to models/')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray, num_leaves: int, learning_rate: float, max_depth: int, n_estimators: int) -> lgb.LGBMClassifier:
    """Train a LightGBM model."""
    try:
        best_model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
            metric="multi_logloss",
            is_unbalance=True,
            class_weight="balanced",
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )
        best_model.fit(X_train, y_train)
        logger.debug('LightGBM model training completed')
        return best_model
    except Exception as e:
        logger.error('Error during LightGBM model training: %s', e)
        raise


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise


def get_root_directory() -> str:
    """Get the root directory (two levels up from this script's location)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['tfidf']['max_features']
        ngram_range = tuple(params['tfidf']['ngram_range'])

        num_leaves = params['model']['num_leaves']
        learning_rate = params['model']['learning_rate']
        max_depth = params['model']['max_depth']
        n_estimators = params['model']['n_estimators']

        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment('sentidex-model-building')

        with mlflow.start_run():
            # Log all hyperparameters
            mlflow.log_params({
                'max_features': max_features,
                'ngram_range': str(ngram_range),
                'num_leaves': num_leaves,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'n_estimators': n_estimators
            })

            # Apply TF-IDF feature engineering on training data
            X_train_tfidf, y_train = apply_tfidf(train_data, max_features, ngram_range)

            # Train the LightGBM model using hyperparameters from params.yaml
            best_model = train_lgbm(X_train_tfidf, y_train, num_leaves, learning_rate, max_depth, n_estimators)

            # Save the trained model to models/ folder
            models_dir = os.path.join(root_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'lgbm_model.pkl')
            save_model(best_model, model_path)

            # Log artifacts to MLflow
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))

            logger.debug('Model building complete. Artifacts saved to models/ and logged to MLflow.')

    except Exception as e:
        logger.error('Failed to complete the feature engineering and model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()