import pytest
from src.app import init_app
from src.models.classifier import FakeNewsClassifier
import json

@pytest.fixture
def app():
    app = init_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def mock_classifier(monkeypatch):
    """Mock classifier for testing"""
    class MockClassifier:
        def predict(self, text):
            if "fake" in text.lower():
                return 0, 0.85
            return 1, 0.92

    monkeypatch.setattr(FakeNewsClassifier, 'load_model', lambda x: MockClassifier())

def test_home_page(client):
    """Test home page loads correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Fake News Detector' in response.data

def test_predict_endpoint_success(client, mock_classifier):
    """Test successful prediction request"""
    data = {'text': 'This is a legitimate news article.'}
    response = client.post('/predict', data=data)
    
    assert response.status_code == 200
    result = json.loads(response.data)
    
    assert result['status'] == 'success'
    assert 'prediction' in result
    assert 'probability' in result
    assert 'class' in result

def test_predict_endpoint_fake_news(client, mock_classifier):
    """Test prediction with fake news text"""
    data = {'text': 'FAKE NEWS! This is completely false!'}
    response = client.post('/predict', data=data)
    
    assert response.status_code == 200
    result = json.loads(response.data)
    
    assert result['status'] == 'success'
    assert result['prediction'] == 'FAKE'
    assert result['class'] == 'fake'

def test_predict_endpoint_empty_text(client):
    """Test prediction with empty text"""
    data = {'text': ''}
    response = client.post('/predict', data=data)
    
    assert response.status_code == 400
    result = json.loads(response.data)
    
    assert result['status'] == 'error'
    assert 'error' in result

def test_predict_endpoint_missing_text(client):
    """Test prediction with missing text field"""
    response = client.post('/predict', data={})
    
    assert response.status_code == 400
    result = json.loads(response.data)
    
    assert result['status'] == 'error'
    assert 'error' in result

def test_404_error(client):
    """Test 404 error handling"""
    response = client.get('/nonexistent-page')
    
    assert response.status_code == 404
    result = json.loads(response.data)
    
    assert result['status'] == 'error'
    assert 'error' in result