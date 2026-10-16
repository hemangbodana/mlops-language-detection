# Language Detection (MLOps Practical)

Flask ML app that detects the language of a text using TF-IDF character n-grams and Multinomial Naive Bayes, trained on WiLI-2018 (235 languages, 94.5% test accuracy).

## Run locally
    pip install -r requirements.txt
    python app.py

## Endpoints
- GET  /         web interface
- GET  /health   health check
- POST /predict  JSON: {"text": "..."} returns detected language and top 3