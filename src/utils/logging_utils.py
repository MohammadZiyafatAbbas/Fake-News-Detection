"""
Configure logging for the Fake News Detection project.
"""

import logging
import logging.config
from pathlib import Path
from src.config import LOGGING_CONFIG, LOGS_DIR

def setup_logging():
    """Set up logging configuration for the project."""
    # Create logs directory if it doesn't exist
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Get the root logger
    logger = logging.getLogger()
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)

# Custom exception classes
class ModelNotFoundError(Exception):
    """Raised when a model file cannot be found."""
    pass

class ModelNotTrainedError(Exception):
    """Raised when trying to use a model that hasn't been trained."""
    pass

class DataValidationError(Exception):
    """Raised when there are issues with input data."""
    pass

class FeatureExtractionError(Exception):
    """Raised when there are issues during feature extraction."""
    pass

# Error handling decorators
def handle_model_errors(func):
    """Decorator to handle model-related errors."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except ModelNotFoundError as e:
            logger.error(f"Model not found: {str(e)}")
            raise
        except ModelNotTrainedError as e:
            logger.error(f"Model not trained: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def handle_data_errors(func):
    """Decorator to handle data-related errors."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        try:
            return func(*args, **kwargs)
        except DataValidationError as e:
            logger.error(f"Data validation error: {str(e)}")
            raise
        except FeatureExtractionError as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            raise
    return wrapper