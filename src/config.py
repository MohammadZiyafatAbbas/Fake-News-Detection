"""
Configuration settings for the Fake News Detection project.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'feature_engineering': {
        'tfidf': {
            'max_features': 50000,
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 0.95
        }
    },
    'training': {
        'test_size': 0.2,
        'random_state': 42,
        'n_splits': 5
    },
    'model_params': {
        'logistic_regression': {
            'C': 1.0,
            'max_iter': 1000,
            'class_weight': 'balanced'
        },
        'random_forest': {
            'n_estimators': 200,
            'max_depth': 20,
            'class_weight': 'balanced'
        },
        'lightgbm': {
            'n_estimators': 200,
            'num_leaves': 31,
            'class_weight': 'balanced'
        }
    }
}

# File paths
TRAIN_DATA_PATH = DATA_DIR / 'train.csv'
VALID_DATA_PATH = DATA_DIR / 'valid.csv'
TEST_DATA_PATH = DATA_DIR / 'test.csv'
MODEL_SAVE_PATH = MODELS_DIR / 'advanced_model.joblib'

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'app.log',
            'formatter': 'standard'
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}

# API configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False
}

# Front-end configuration
FRONTEND_CONFIG = {
    'template_dir': PROJECT_ROOT / 'templates',
    'static_dir': PROJECT_ROOT / 'static'
}