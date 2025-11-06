import logging
from typing import Tuple, Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from advanced_classifier import AdvancedFakeNewsClassifier
from src.utils.data_utils import load_dataset
from src.config import MODEL_SAVE_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load datasets using the project's data utility and normalize labels.

    Returns train, valid, test DataFrames with invalid label rows dropped.
    """
    logger.info("Loading datasets via src.utils.data_utils.load_dataset...")
    train_df = load_dataset('train.csv')
    valid_df = load_dataset('valid.csv')
    test_df = load_dataset('test.csv')

    # Drop rows with unresolved/missing labels
    for name, df in (('train', train_df), ('valid', valid_df), ('test', test_df)):
        if 'Label' not in df.columns:
            logger.error(f"{name} dataset missing 'Label' column")
            raise ValueError(f"{name} dataset missing 'Label' column")

        before = len(df)
        df.dropna(subset=['Label'], inplace=True)
        after = len(df)
        if after < before:
            logger.info(f"Dropped {before-after} rows with invalid labels from {name} set")

    return train_df, valid_df, test_df

def plot_evaluation_metrics(results: Dict) -> None:
    """Plot evaluation metrics for all models."""
    models = list(results.keys())
    cv_means = [results[model]['cv_mean'] for model in models]
    val_scores = [results[model]['val_score'] for model in models]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, cv_means, width, label='CV Score')
    plt.bar(x + width/2, val_scores, width, label='Validation Score')
    
    plt.xlabel('Models')
    plt.ylabel('ROC-AUC Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_confusion_matrix(cm: np.ndarray) -> None:
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load and preprocess data
    train_df, valid_df, test_df = load_and_preprocess_data()
    
    # Initialize classifier
    classifier = AdvancedFakeNewsClassifier()
    
    # Train and evaluate models
    logger.info("Training and evaluating models...")
    results = classifier.train_evaluate_models(
        train_df['Statement'], train_df['Label'],
        valid_df['Statement'], valid_df['Label']
    )
    
    # Plot evaluation metrics
    plot_evaluation_metrics(results)
    
    # Detailed evaluation on test set
    logger.info("Performing detailed evaluation on test set...")
    test_evaluation = classifier.detailed_evaluation(test_df['Statement'], test_df['Label'])
    
    # Plot confusion matrix
    plot_confusion_matrix(test_evaluation['confusion_matrix'])
    
    # Log detailed results
    logger.info("\nTest Set Classification Report:")
    logger.info(test_evaluation['classification_report'])
    logger.info(f"\nTest Set ROC-AUC Score: {test_evaluation['roc_auc']:.4f}")
    
    # Save the best model
    if classifier.best_model is not None:
        # MODEL_SAVE_PATH is a pathlib.Path
        classifier.save_model(str(MODEL_SAVE_PATH))
        logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        logger.warning("No model was selected as best_model; nothing saved.")

    logger.info("Model training and evaluation completed!")

if __name__ == "__main__":
    main()