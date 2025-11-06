document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('detection-form');
    const newsText = document.getElementById('news-text');
    const analyzeBtn = document.getElementById('analyze-btn');
    const clearBtn = document.getElementById('clear-btn');
    const resultDiv = document.getElementById('result');
    const errorDiv = document.getElementById('error');
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceValue = document.getElementById('confidence-value');

    // Clear form handler
    clearBtn.addEventListener('click', function() {
        newsText.value = '';
        resultDiv.style.display = 'none';
        errorDiv.style.display = 'none';
        enableForm(true);
    });

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const text = newsText.value.trim();
        
        if (!text) {
            showError('Please enter some text to analyze.');
            return;
        }
        
        try {
            enableForm(false);
            showLoading(true);
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `text=${encodeURIComponent(text)}`
            });
            
            const data = await response.json();
            
            if (data.status === 'success') {
                showResult(data);
            } else {
                showError(data.error || 'An error occurred during analysis.');
            }
        } catch (error) {
            showError('Network error occurred. Please try again.');
        } finally {
            showLoading(false);
            enableForm(true);
        }
    });

    // Helper functions
    function showResult(data) {
        errorDiv.style.display = 'none';
        resultDiv.style.display = 'block';
        
        // Update prediction label
        predictionLabel.textContent = data.prediction;
        predictionLabel.className = data.class === 'fake' ? 'text-danger' : 'text-success';
        
        // Update confidence bar
        confidenceBar.style.width = `${data.probability}%`;
        confidenceBar.className = `progress-bar ${data.class === 'fake' ? 'bg-danger' : 'bg-success'}`;
        
        // Update confidence value
        confidenceValue.textContent = `${data.probability}%`;
        
        // Add fade-in animation
        resultDiv.classList.add('fade-in');
        setTimeout(() => resultDiv.classList.remove('fade-in'), 500);
        
        // Update alert class
        const alertDiv = resultDiv.querySelector('.alert');
        alertDiv.className = `alert ${data.class === 'fake' ? 'alert-danger' : 'alert-success'}`;
    }

    function showError(message) {
        resultDiv.style.display = 'none';
        errorDiv.style.display = 'block';
        document.getElementById('error-message').textContent = message;
        
        // Add fade-in animation
        errorDiv.classList.add('fade-in');
        setTimeout(() => errorDiv.classList.remove('fade-in'), 500);
    }

    function enableForm(enable) {
        newsText.disabled = !enable;
        analyzeBtn.disabled = !enable;
        clearBtn.disabled = !enable;
    }

    function showLoading(show) {
        if (show) {
            // Remove any existing spinner
            const existingSpinner = document.querySelector('.spinner');
            if (existingSpinner) existingSpinner.remove();
            
            // Create and add new spinner
            const spinner = document.createElement('div');
            spinner.className = 'spinner';
            analyzeBtn.insertAdjacentElement('afterend', spinner);
        } else {
            // Remove spinner
            const spinner = document.querySelector('.spinner');
            if (spinner) spinner.remove();
        }
    }
    
    // Text area auto-resize
    newsText.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
});