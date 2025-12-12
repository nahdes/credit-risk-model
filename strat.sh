#!/bin/bash
# Create GitHub workflows directory and CI/CD file
mkdir -p .github/workflows
touch .github/workflows/ci.yml
echo "# For CI/CD" >> .github/workflows/ci.yml

# Create data directories
mkdir -p data/raw
mkdir -p data/processed
echo "# add this folder to .gitignore" >> data/.gitignore
echo "# Raw data goes here" >> data/raw/README.md
echo "# Processed data for training" >> data/processed/README.md

# Create notebooks directory and EDA notebook
mkdir -p notebooks
touch notebooks/eda.ipynb
echo "# Exploratory, one-off analysis" >> notebooks/eda.ipynb

# Create source code directory structure
mkdir -p src/api

# Create Python files in src/
touch src/__init__.py
touch src/data_processing.py
touch src/train.py
touch src/predict.py
touch src/api/main.py
touch src/api/pydantic_models.py

# Add comments to Python files
echo "# Script for feature engineering" >> src/data_processing.py
echo "# Script for model training" >> src/train.py
echo "# Script for inference" >> src/predict.py
echo "# FastAPI application" >> src/api/main.py
echo "# Pydantic models for API" >> src/api/pydantic_models.py

# Create tests directory and test file
mkdir -p tests
touch tests/test_data_processing.py
echo "# Unit tests" >> tests/test_data_processing.py

# Create root-level files
touch Dockerfile
touch docker-compose.yml
touch requirements.txt
touch .gitignore
touch README.md

echo "✅ Credit risk model folder structure created successfully!"
echo "📁 Project root: $(pwd)"