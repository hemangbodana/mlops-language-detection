import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
    assert r.get_json()["languages_supported"] == 235


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Language Detection" in r.data


def test_predict_english(client):
    r = client.post("/predict", json={"text": "Hello, how are you doing today?"})
    assert r.status_code == 200
    assert r.get_json()["detected"]["code"] == "eng"


def test_predict_french(client):
    r = client.post("/predict", json={"text": "Bonjour, comment allez-vous aujourd'hui ?"})
    assert r.get_json()["detected"]["code"] == "fra"


def test_predict_returns_top3(client):
    r = client.post("/predict", json={"text": "Wie geht es dir heute?"})
    assert len(r.get_json()["top3"]) == 3


def test_predict_missing_text(client):
    r = client.post("/predict", json={})
    assert r.status_code == 400


def test_predict_too_short(client):
    r = client.post("/predict", json={"text": "hi"})
    assert r.status_code == 400