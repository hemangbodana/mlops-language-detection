import time
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

DATA_DIR = "wili"

def load(x_path, y_path):
    with open(x_path, encoding="utf-8") as f:
        texts = f.read().splitlines()
    with open(y_path, encoding="utf-8") as f:
        langs = f.read().splitlines()
    return pd.DataFrame({"text": texts, "language": langs})

train = load(f"{DATA_DIR}/x_train.txt", f"{DATA_DIR}/y_train.txt")
test = load(f"{DATA_DIR}/x_test.txt", f"{DATA_DIR}/y_test.txt")

model = make_pipeline(
    TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        max_features=60000,
        sublinear_tf=True,
    ),
    MultinomialNB(alpha=0.01),
)

start = time.time()
model.fit(train["text"], train["language"])
print(f"Training time: {time.time() - start:.1f} s")

pred = model.predict(test["text"])
print(f"Accuracy : {accuracy_score(test['language'], pred):.4f}")
print(f"Macro F1 : {f1_score(test['language'], pred, average='macro'):.4f}")

# Weakest 10 languages (useful for the report's analysis)
rep = classification_report(test["language"], pred, output_dict=True, zero_division=0)
rows = {k: v for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")}
worst = sorted(rows.items(), key=lambda kv: kv[1]["f1-score"])[:10]
print("\nLowest F1 languages:")
for lang, m in worst:
    print(f"  {lang:>5}  F1={m['f1-score']:.3f}")

joblib.dump(model, "model.pkl", compress=3)
print("\nSaved model.pkl")
