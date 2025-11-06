import pytest
from src.models.classifier import FakeNewsClassifier
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

@pytest.fixture
def classifier():
    return FakeNewsClassifier()

@pytest.fixture
def sample_data():
    texts = [
        "This is a legitimate news article about important events.",
        "FAKE NEWS! Everything is a conspiracy! Click here to learn more!",
        "Scientists discover new species in the Amazon rainforest.",
    ]
    labels = [1, 0, 1]  # 1 for real, 0 for fake
    return pd.Series(texts), pd.Series(labels)

def test_classifier_initialization(classifier):
    """Test if classifier initializes correctly"""
    assert classifier is not None
    assert classifier.best_model is None
    assert classifier.feature_pipeline is None

def test_create_feature_pipeline(classifier):
    """Test feature pipeline creation"""
    classifier.create_feature_pipeline()
    assert classifier.feature_pipeline is not None

def test_create_models(classifier):
    """Test model creation"""
    classifier.create_models()
    assert len(classifier.models) > 0
    assert 'logistic_regression' in classifier.models
    assert 'random_forest' in classifier.models

def test_extract_nlp_features(classifier):
    """Test NLP feature extraction"""
    text = "This is a test article about important news."
    features = classifier.extract_nlp_features(text)
    
    assert isinstance(features, dict)
    assert 'num_entities' in features
    assert 'text_length' in features
    assert features['text_length'] == len(text)

@pytest.mark.parametrize("text,expected_type", [
    ("Real news article", 1),
    ("FAKE NEWS!!! Click here!", 0)
])
def test_predict(classifier, text, expected_type):
    """Test prediction functionality"""
    # Mock the feature pipeline and model
    classifier.feature_pipeline = MagicMock()
    classifier.best_model = MagicMock()
    
    # Set up mock returns
    classifier.feature_pipeline.transform.return_value = np.array([[1, 2, 3]])
    classifier.best_model.predict.return_value = np.array([expected_type])
    classifier.best_model.predict_proba.return_value = np.array([[0.2, 0.8]])
    
    prediction, probability = classifier.predict(text)
    
    assert prediction == expected_type
    assert 0 <= probability <= 1

def test_train_evaluate_models(classifier, sample_data):
    """Test model training and evaluation"""
    X, y = sample_data
    X_val = X.copy()
    y_val = y.copy()
    
    classifier.create_feature_pipeline()
    classifier.create_models()
    
    results = classifier.train_evaluate_models(X, y, X_val, y_val)
    
    assert isinstance(results, dict)
    assert len(results) > 0
    assert all('cv_mean' in model_result for model_result in results.values())
    assert classifier.best_model is not None

def test_save_load_model(classifier, tmp_path):
    """Test model saving and loading"""
    # Create and configure a model
    classifier.create_feature_pipeline()
    classifier.create_models()
    classifier.best_model = classifier.models['logistic_regression']
    
    # Save the model
    save_path = tmp_path / "test_model.joblib"
    classifier.save_model(str(save_path))
    
    # Load the model
    loaded_classifier = FakeNewsClassifier.load_model(str(save_path))
    
    assert loaded_classifier.best_model is not None
    assert loaded_classifier.feature_pipeline is not None

def test_detailed_evaluation(classifier, sample_data):
    """Test detailed model evaluation"""
    X, y = sample_data
    
    # Mock the feature pipeline and model
    classifier.feature_pipeline = MagicMock()
    classifier.best_model = MagicMock()
    
    # Set up mock returns
    classifier.feature_pipeline.transform.return_value = np.array([[1, 2, 3]])
    classifier.best_model.predict.return_value = np.array([1, 0, 1])
    classifier.best_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    
    evaluation = classifier.detailed_evaluation(X, y)
    
    assert isinstance(evaluation, dict)
    assert 'classification_report' in evaluation
    assert 'confusion_matrix' in evaluation
    assert 'roc_auc' in evaluation