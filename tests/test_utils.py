import pytest
import pandas as pd
import numpy as np
from src.utils.data_utils import (
    load_dataset,
    extract_text_features,
    evaluate_predictions,
    save_evaluation_results
)
import json
import os

@pytest.fixture
def sample_csv(tmp_path):
    """Create a sample CSV file for testing"""
    df = pd.DataFrame({
        'Statement': ['True news', 'Fake news', 'More true news'],
        'Label': ['true', 'false', 'true']
    })
    filepath = tmp_path / "test.csv"
    df.to_csv(filepath, index=False)
    return filepath

@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing"""
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_prob = np.array([0.8, 0.2, 0.4, 0.3, 0.9])
    return y_true, y_pred, y_prob

def test_load_dataset(sample_csv):
    """Test dataset loading functionality"""
    df = load_dataset(sample_csv)
    
    assert isinstance(df, pd.DataFrame)
    assert 'Statement' in df.columns
    assert 'Label' in df.columns
    assert df['Label'].isin([0, 1]).all()

def test_load_dataset_error():
    """Test dataset loading with nonexistent file"""
    with pytest.raises(Exception):
        load_dataset("nonexistent.csv")

def test_extract_text_features():
    """Test text feature extraction"""
    text = "This is a TEST article! With some punctuation..."
    features = extract_text_features(text)
    
    assert isinstance(features, dict)
    assert features['length'] == len(text)
    assert features['word_count'] == 8
    assert features['punctuation_count'] == 3
    assert 0 <= features['capital_ratio'] <= 1

def test_extract_text_features_empty():
    """Test feature extraction with empty text"""
    features = extract_text_features("")
    
    assert isinstance(features, dict)
    assert features['length'] == 0
    assert features['word_count'] == 0
    assert features['punctuation_count'] == 0
    assert features['capital_ratio'] == 0

def test_evaluate_predictions(sample_predictions):
    """Test prediction evaluation metrics"""
    y_true, y_pred, y_prob = sample_predictions
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1

def test_save_evaluation_results(tmp_path, sample_predictions):
    """Test saving evaluation results"""
    y_true, y_pred, y_prob = sample_predictions
    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    
    filepath = tmp_path / "results.json"
    save_evaluation_results(metrics, filepath)
    
    assert os.path.exists(filepath)
    
    with open(filepath, 'r') as f:
        loaded_results = json.load(f)
    
    assert loaded_results == metrics