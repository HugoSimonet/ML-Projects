.PHONY: help install test clean train api docker

help:
	@echo "Recommendation System - Available Commands"
	@echo "==========================================="
	@echo "install       Install dependencies"
	@echo "install-dev   Install dev dependencies"
	@echo "test          Run tests"
	@echo "test-cov      Run tests with coverage"
	@echo "train         Train models on synthetic data"
	@echo "train-ml      Train models on MovieLens"
	@echo "api           Start API server"
	@echo "clean         Clean generated files"
	@echo "docker-build  Build Docker image"
	@echo "docker-up     Start Docker containers"
	@echo "docker-down   Stop Docker containers"
	@echo "lint          Run code linting"
	@echo "format        Format code"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov black isort flake8 mypy

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

train:
	python scripts/train_models.py --dataset synthetic --models all --save --visualize

train-ml:
	python scripts/train_models.py --dataset movielens --version 100k --models all --save --visualize

quick-start:
	python examples/quick_start.py

api:
	python -m src.api.app

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "API running at http://localhost:8000"
	@echo "Docs available at http://localhost:8000/docs"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	isort src/ tests/ scripts/

setup-data:
	mkdir -p data/raw data/processed data/external models logs

.DEFAULT_GOAL := help
