# Language Detection: MLOps Practical Project

A machine learning web application that detects the language of any text. It uses TF-IDF character n-grams with a Multinomial Naive Bayes classifier, trained on the WiLI-2018 dataset (235 languages). The model is served through a Flask app, packaged with Docker, and deployed on AWS EC2 and Google Cloud.

**Model performance (WiLI-2018 test set, 117,500 samples):** accuracy 94.51%, macro F1 0.9493.

## Technologies and Tools Used

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn 1.6.1 (TfidfVectorizer, MultinomialNB), NumPy, joblib |
| Dataset | WiLI-2018 (Wikipedia Language Identification) |
| Training | Google Colab |
| Web app | Flask, HTML/CSS, Gunicorn |
| Testing | pytest |
| Containerization | Docker |
| Cloud | AWS EC2 (Amazon Linux), Google Cloud Platform |
| Version control | Git, GitHub |

## Project Structure

```
mlops-language-detection/
├── app.py              # Flask application
├── train_model.py      # Model training script (run in Colab)
├── model.pkl           # Trained TF-IDF + Naive Bayes pipeline
├── labels.csv          # Language code to name mapping
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── templates/index.html
├── static/style.css
└── tests/test_app.py
```

## Instructions to Run the Project

### 1. Run locally

```bash
git clone https://github.com/hemangbodana/mlops-language-detection.git
cd mlops-language-detection
python -m venv venv
venv\Scripts\activate        # Windows (use: source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 in your browser.

### 2. Run with Docker

```bash
docker build -t language-detection:1.0 .
docker run -d --name lang-app -p 5000:5000 language-detection:1.0
```

Open http://localhost:5000. Stop and remove it with `docker rm -f lang-app`.

### 3. Run the tests

```bash
python -m pytest -v
```

### 4. Retrain the model (optional)

Download WiLI-2018 from https://zenodo.org/records/841984, place the extracted files in a `wili/` folder, and run `python train_model.py` (originally run in Google Colab).

## API Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/` | Web interface |
| GET | `/health` | Health check, returns model status and number of languages |
| POST | `/predict` | JSON body `{"text": "..."}`, returns the detected language and top 3 predictions |

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Wie geht es dir heute?"}'
```

Text must be between 3 and 5000 characters. Invalid input returns HTTP 400.

## Deployment URLs

| Platform | URL |
|---|---|
| AWS EC2 (ap-southeast-2) | http://3.26.76.215:5000 |
| Google Cloud Platform | http://YOUR_GCP_IP:5000 |

Health checks: `/health` on each URL above.

> Note: the cloud instances may be stopped to avoid charges, and the public IP changes after a restart. If a link does not respond, the instance is offline.

## Author

Hemang Bodana
