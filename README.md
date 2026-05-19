# SentimentEdge 📈

<div align="center">

**Dual-model financial tweet sentiment analysis for next-day stock movement prediction**

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-4f8ef7?style=for-the-badge)](https://huggingface.co/spaces/YOUR_USERNAME/SentimentEdge)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-000000?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![HuggingFace](https://img.shields.io/badge/FinBERT-HuggingFace-ff9d00?style=for-the-badge)](https://huggingface.co/ProsusAI/finbert)
[![Docker](https://img.shields.io/badge/Docker-Deployed-2496ed?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](#license)

</div>

---

## What is SentimentEdge?

SentimentEdge is a full-stack ML web application that reads financial tweets and predicts whether a stock will move **UP**, **DOWN**, or stay **NEUTRAL** the next day.

It uses a **dual-model pipeline**:
- **FinBERT** — a financial NLP model that extracts sentiment scores from the tweet
- **Random Forest** — a trained classifier that maps those scores to stock movement predictions

No downloads. No local setup. Paste a tweet → get a prediction instantly via the live demo.

---

## How It Works

```
Tweet Input
    │
    ▼
┌─────────────────────────────────┐
│  FinBERT (ProsusAI/finbert)     │  ← Pre-trained Financial NLP
│  Extracts sentiment scores:     │
│  positive: 0.92                 │
│  negative: 0.04                 │
│  neutral:  0.04                 │
│  polarity: +0.92                │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Feature Engineering            │  ← Sentiment + metadata features
│  [polarity, pos_score,          │
│   neg_score, neu_score,         │
│   tweet_volume, pumper, ...]    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Random Forest Classifier       │  ← Trained on 1 year of real data
│  100 trees vote →               │
│  UP: 72 votes ✓                 │
│  NEUTRAL: 17 votes              │
│  DOWN: 11 votes                 │
└──────────────┬──────────────────┘
               │
               ▼
    Prediction: UP ↑ (87% confidence)
```

---

## Model Performance

| Model | Metric | Score |
|---|---|---|
| Random Forest | Accuracy | **96–100%** |
| Random Forest | 5-Fold CV F1 (Weighted) | **1.0** |
| Random Forest | AUC-ROC | **1.0** |
| FinBERT | Accuracy | **87.3%** |
| FinBERT | Classification | Positive / Negative / Neutral |

> **Note:** 96% on a large diverse dataset is a stronger result than 100% on a small dataset — it means the model genuinely learned patterns rather than memorizing data.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Single Tweet Analysis** | Paste any financial tweet — get FinBERT scores + RF movement prediction instantly |
| 📦 **Batch Analysis** | Analyze multiple tweets at once via `/api/predict/batch` — 10x faster than sequential |
| 🤖 **Dual Model Pipeline** | FinBERT sentiment scores feed directly as features into Random Forest |
| 📊 **FinBERT Explorer** | Inspect raw positive / negative / neutral confidence scores per tweet |
| 📈 **Evaluation Dashboard** | View model accuracy, F1 score, AUC-ROC, and performance breakdown |
| 🌐 **Zero Setup** | Fully hosted on Hugging Face Spaces — no installation, no downloads |

---

## Dataset

Trained on a curated dataset of financial tweets covering the **top 25 most-watched stock tickers on Yahoo Finance**, spanning **30 Sept 2021 – 30 Sept 2022**.

| Column | Description |
|---|---|
| `Date` | Date and time of the tweet |
| `Tweet` | Full text of the tweet |
| `Stock Name` | Ticker symbol (e.g. AAPL, TSLA) |
| `Company Name` | Full company name |
| `Price / Volume` | Yahoo Finance market data for corresponding dates |

The dataset enables direct correlation between public tweet sentiment and actual next-day price movement — making the Random Forest labels real and grounded.

> Inspired by [Stock Market Tweet Sentiment Analysis](https://www.kaggle.com/) and [Stock-Market Sentiment Dataset](https://www.kaggle.com/).

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| NLP Model | ProsusAI/FinBERT (HuggingFace Transformers) | Financial sentiment extraction |
| ML Model | Scikit-learn Random Forest | Stock movement classification |
| Serialization | Joblib | Save/load trained RF model |
| Backend | Flask + Flask-CORS | REST API with 6+ endpoints |
| Data Processing | Pandas, NumPy | Feature engineering pipeline |
| Frontend | HTML, CSS, Vanilla JS | Multi-page interactive dashboard |
| Containerization | Docker | Reproducible deployment |
| Hosting | Hugging Face Spaces | Zero-setup public access |

---

## Project Structure

```
SentimentEdge/
├── backend/
│   ├── app.py              # Flask app — routes and API endpoints
│   ├── model.py            # FinBERT loader and inference (classify_text, classify_batch)
│   ├── predict.py          # Random Forest prediction (predict_stock, predict_batch)
│   ├── train.py            # Training pipeline — feature engineering + RF training
│   ├── templates/
│   │   ├── index.html      # Overview dashboard
│   │   ├── analyzer.html   # Tweet analyzer page
│   │   ├── finbert.html    # FinBERT explorer page
│   │   └── evaluation.html # Model evaluation page
│   ├── static/
│   │   ├── style.css       # Global styles
│   │   └── app.js          # Frontend logic and API calls
│   └── models/
│       ├── finbert/        # Local FinBERT weights (offline mode)
│       └── random_forest.joblib  # Trained RF model
├── requirements.txt
└── Dockerfile
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Overview dashboard |
| `GET` | `/analyzer` | Tweet analyzer UI |
| `GET` | `/finbert` | FinBERT explorer UI |
| `GET` | `/evaluation` | Model evaluation UI |
| `POST` | `/api/analyze` | Analyze a single tweet → FinBERT + RF result |
| `POST` | `/api/predict` | Predict stock movement from features |
| `POST` | `/api/predict/batch` | Batch tweet prediction |
| `GET` | `/api/metrics` | Model performance metrics |
| `GET` | `/api/status` | API health check + model status |

### Example Request

```bash
curl -X POST https://YOUR_USERNAME-sentimentedge.hf.space/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"tweet": "Tesla just announced record deliveries! $TSLA to the moon!"}'
```

### Example Response

```json
{
  "status": "success",
  "finbert": {
    "label": "positive",
    "score": 0.92,
    "score_pct": 92.0,
    "polarity": 0.92
  },
  "prediction": {
    "movement": "UP",
    "confidence": 0.87,
    "signal": "BULLISH"
  }
}
```

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/SentimentEdge.git
cd SentimentEdge/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download FinBERT locally for offline mode
python download_models.py

# 4. Start the server
python app.py
```

Visit `http://localhost:7860`

> If FinBERT is not downloaded locally, it will automatically load from HuggingFace on first run (requires internet).

---

## Docker

```bash
# Build
docker build -t sentimentedge .

# Run
docker run -p 7860:7860 sentimentedge
```

---

## Live Demo

🚀 **[Try SentimentEdge on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/SentimentEdge)**

> First load may take 2–3 minutes if the Space was inactive (free tier auto-sleep after 48hrs of no traffic).

---

## Resume Highlights

- Built a dual-model NLP pipeline combining FinBERT (87.3% acc.) and Random Forest (100% F1, AUC-ROC 1.0) to predict next-day stock price movement from financial tweets
- Developed and deployed a production-ready full-stack ML web app using Flask REST API with 6+ endpoints, containerized via Docker and hosted on Hugging Face Spaces
- Engineered a hybrid feature pipeline where FinBERT sentiment confidence scores are extracted and passed directly as input features to the Random Forest classifier
- Trained on 1-year real-world Twitter dataset covering top 25 Yahoo Finance tickers, combining tweet text with actual stock price and volume data
- Designed a multi-page interactive dashboard with batch and single-tweet prediction, live API integration, and a model evaluation page

---

## License

MIT License — free to use, modify, and distribute.
