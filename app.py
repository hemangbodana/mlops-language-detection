import csv
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
LABELS_PATH = os.path.join(BASE_DIR, "labels.csv")

MIN_CHARS = 3
MAX_CHARS = 5000

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def load_language_names():
    """Map WiLI label codes (e.g. 'fra') to English names (e.g. 'French')."""
    try:
        with open(LABELS_PATH, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=";")
            return {row["Label"]: row["English"] for row in reader}
    except Exception:
        return {}


LANG_NAMES = load_language_names()


def validate_text(text):
    """Return (clean_text, error_message)."""
    if not isinstance(text, str):
        return None, "Field 'text' is required and must be a string."
    text = text.strip()
    if len(text) < MIN_CHARS:
        return None, f"Text must be at least {MIN_CHARS} characters long."
    if len(text) > MAX_CHARS:
        return None, f"Text must be at most {MAX_CHARS} characters long."
    return text, None


def predict_language(text):
    probs = model.predict_proba([text])[0]
    top_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {
            "code": str(model.classes_[i]),
            "language": LANG_NAMES.get(model.classes_[i], str(model.classes_[i])),
            "confidence": round(float(probs[i]) * 100, 2),
        }
        for i in top_idx
    ]
    return {"detected": top3[0], "top3": top3}


@app.route("/", methods=["GET", "POST"])
def index():
    result, error, text = None, None, ""
    if request.method == "POST":
        text, error = validate_text(request.form.get("text"))
        if error is None:
            result = predict_language(text)
        text = request.form.get("text", "")
    return render_template("index.html", result=result, error=error, text=text)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        status="ok",
        model_loaded=model is not None,
        languages_supported=len(model.classes_),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text, error = validate_text(data.get("text"))
    if error:
        return jsonify(error=error), 400
    return jsonify(predict_language(text))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)