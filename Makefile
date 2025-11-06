.PHONY: install test lint format clean docs

# Python interpreter to use
PYTHON := python
PIP := pip

# Install all dependencies
install:
	$(PIP) install -e ".[dev]"

# Run tests with coverage
test:
	pytest tests/ --cov=src --cov-report=term-missing

# Run linting
lint:
	flake8 src/ tests/
	mypy src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.pyc" -exec rm -rf {} +
	find . -type d -name "*.pyo" -exec rm -rf {} +
	find . -type d -name "*.pyd" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

# Build documentation
docs:
	cd docs && make html