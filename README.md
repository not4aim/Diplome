## README.md English version

# Diplome – ML System for Data Engineering

## Description
Software system for data engineering in machine learning: intelligent data analysis and neural networks.  
The project implements the full data workflow: **ETL → ML → API → UI → DB**.

## Features
- ETL module: data cleaning, transformation, feature engineering.
- ML models: RandomForest and Neural Network.
- REST API (FastAPI): `/etl`, `/train`, `/predict`, `/predict_batch_file`, `/metrics`.
- Web interface (HTML+JS) for user interaction.
- Metrics: accuracy, precision, recall, f1, confusion matrix.
- Docker infrastructure for deployment.

## Run (main scenario)
### Docker
```bash
docker-compose up --build

API will be available at: http://localhost:8000

## Stop project
'''bash
  docker-compose down

## Restart project
'''bash
docker-compose down
docker-compose up --build

## Local run (for testing)
'''bash
pip install -r requirements.txt
uvicorn api:app --reload