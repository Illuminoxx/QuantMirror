# ═══════════════════════════════════════════════════════════════
#  predict.py  —  SentimentEdge
#
#  FIX: FinBERT NEGATIVE at 94% should never produce RISE.
#  Resolution logic:
#    1. FinBERT confidence >= FINBERT_OVERRIDE_THRESHOLD (85%)
#       → FinBERT wins regardless of RF output
#    2. RF confidence >= RF_CONFIDENCE_THRESHOLD (70%)
#       → RF wins (it has enough confidence in its own prediction)
#    3. Both uncertain → weighted blend of both signals
# ═══════════════════════════════════════════════════════════════

import os
import re
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "rf_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# ── Confidence thresholds ──
FINBERT_OVERRIDE_THRESHOLD = 0.85   # FinBERT wins if score >= this
RF_CONFIDENCE_THRESHOLD    = 0.70   # RF wins if prob >= this

_rf_model   = None
_rf_metrics = None


def get_model():
    global _rf_model
    if _rf_model is None and os.path.exists(MODEL_PATH):
        print("[predict] Loading RF model...")
        _rf_model = joblib.load(MODEL_PATH)
        print("[predict] RF model loaded.")
    return _rf_model


def get_metrics():
    global _rf_metrics
    if _rf_metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _rf_metrics = json.load(f)
    return _rf_metrics


# ════════════════════════════════════════════════════
#  FEATURE BUILDER
#  Matches exactly what train.py step 4 produced
# ════════════════════════════════════════════════════
def build_feature_vector(
    tweet_text:        str,
    human_sentiment:   float = 0.0,
    has_human_label:   float = 0.0,
    known_pumper:      float = 0.0,
    price_region:      float = 0.0,
    inflection_point:  float = 0.0,
    tweet_volume:      float = 0.0,
    hour:              int   = 12,
    dayofweek:         int   = 0,
    stock:             str   = "UNKNOWN",
    history_polarities:list  = None,
) -> pd.DataFrame:

    is_weekend       = 1.0 if dayofweek >= 5 else 0.0
    is_market_hours  = 1.0 if (9 <= hour <= 16 and not is_weekend) else 0.0

    recent = history_polarities[-20:] if history_polarities else [human_sentiment]

    def roll(n):
        vals = recent[-n:] if len(recent) >= n else recent
        return float(np.mean(vals)) if vals else 0.0

    human_diff = (human_sentiment - recent[-2]) if len(recent) >= 2 else 0.0

    text = str(tweet_text)
    hs   = float(np.clip(human_sentiment, -1, 1))

    features = {
        # ── Human/derived sentiment (carries the FinBERT signal) ──
        "human_sentiment":    hs,
        "has_human_label":    float(has_human_label),
        "known_pumper":       float(known_pumper),
        "price_region":       float(price_region),
        "inflection_point":   float(inflection_point),
        "tweet_volume":       float(tweet_volume),

        # ── Rolling averages ──
        "human_roll_3":       roll(3),
        "human_roll_7":       roll(7),
        "human_roll_20":      roll(20),
        "human_diff":         float(human_diff),

        # ── Raw text statistics ──
        "tweet_length":       float(len(text)),
        "word_count":         float(len(text.split())),
        "exclaim_count":      float(text.count("!")),
        "question_count":     float(text.count("?")),
        "caps_ratio":         float(len(re.findall(r"[A-Z]", text)) / max(len(text), 1)),
        "has_cashtag":        float(bool(re.search(r"\$[A-Z]+", text))),
        "has_url":            float("http" in text.lower()),
        "has_number":         float(bool(re.search(r"\d+", text))),

        # ── Time ──
        "hour":               float(hour),
        "dayofweek":          float(dayofweek),
        "is_weekend":         is_weekend,
        "is_market_hours":    is_market_hours,

        # ── Technical proxies — derived from human_sentiment, NOT fb_polarity ──
        "rsi_proxy":          float(np.clip(50 + hs * 25, 25, 75)),
        "macd_proxy":         float(hs * 0.4),
        "sma_ratio_proxy":    float(1.0 + hs * 0.03),
        "volume_proxy":       float(abs(hs) * 0.08),
    }

    # ── Stock dummies ──
    metrics = get_metrics()
    if metrics and "feature_names" in metrics:
        for fname in metrics["feature_names"]:
            if fname.startswith("stk_"):
                clean_stock = re.sub(r"[^A-Za-z0-9]", "_", stock).upper()[:12]
                stk_name    = fname[4:]
                features[fname] = 1.0 if clean_stock == stk_name else 0.0

    # ── Build DataFrame in exact training column order ──
    if metrics and "feature_names" in metrics:
        ordered = {k: features.get(k, 0.0) for k in metrics["feature_names"]}
        return pd.DataFrame([ordered])

    return pd.DataFrame([features])


# ════════════════════════════════════════════════════
#  CORE PREDICTION WITH CONFLICT RESOLUTION
# ════════════════════════════════════════════════════
def predict_stock(
    tweet_text:        str,
    stock:             str   = "UNKNOWN",
    human_sentiment:   float = None,
    known_pumper:      float = 0.0,
    price_region:      float = 0.0,
    inflection_point:  float = 0.0,
    tweet_volume:      float = 0.0,
    hour:              int   = None,
    dayofweek:         int   = None,
    history:           list  = None,
) -> dict:

    import datetime
    from model import classify_text

    # ── Step 1: FinBERT sentiment ──
    fb          = classify_text(tweet_text)
    fb_label    = fb["label"]       # "positive" / "neutral" / "negative"
    fb_score    = fb["score"]       # 0.0 → 1.0
    fb_score_pct= fb["score_pct"]
    fb_polarity = fb["polarity"]    # +score, -score, or 0

    # ── Step 2: Resolve human_sentiment ──
    # Derive from FinBERT if not provided by caller
    if human_sentiment is None:
        if fb_label == "positive":   human_sentiment = fb_score       # e.g. +0.943
        elif fb_label == "negative": human_sentiment = -fb_score      # e.g. -0.943
        else:                        human_sentiment = 0.0
        has_human_label = 0.0
    else:
        has_human_label = 1.0

    # ── Step 3: Time defaults ──
    now = datetime.datetime.now()
    if hour      is None: hour      = now.hour
    if dayofweek is None: dayofweek = now.weekday()

    # ── Step 4: Build features ──
    X = build_feature_vector(
        tweet_text        = tweet_text,
        human_sentiment   = human_sentiment,
        has_human_label   = has_human_label,
        known_pumper      = known_pumper,
        price_region      = price_region,
        inflection_point  = inflection_point,
        tweet_volume      = tweet_volume,
        hour              = hour,
        dayofweek         = dayofweek,
        stock             = stock,
        history_polarities= history or [],
    )

    # ── Step 5: RF prediction ──
    rf = get_model()

    if rf is None:
        # No model yet — use FinBERT directly
        is_rise    = fb_polarity >= 0
        confidence = int(fb_score * 100)
        return _build_result(
            prediction   = "Rise" if is_rise else "Fall",
            confidence   = confidence,
            prob_rise    = fb_score if is_rise else (1 - fb_score),
            prob_fall    = (1 - fb_score) if is_rise else fb_score,
            fb_label     = fb_label,
            fb_score     = fb_score,
            fb_score_pct = fb_score_pct,
            fb_polarity  = fb_polarity,
            model_ready  = False,
            decision_src = "FinBERT only (RF not trained yet)",
            stock        = stock,
            features     = list(X.columns),
        )

    pred       = rf.predict(X)[0]           # 0 or 1
    proba      = rf.predict_proba(X)[0]     # [prob_fall, prob_rise]
    prob_rise  = round(float(proba[1]), 4)
    prob_fall  = round(float(proba[0]), 4)
    rf_conf    = max(prob_rise, prob_fall)
    rf_label   = "Rise" if pred == 1 else "Fall"

    # ── Step 6: CONFLICT RESOLUTION ──
    # This is the key fix — resolves NEGATIVE sentiment + RISE prediction conflict
    fb_says_rise = fb_label == "positive"
    fb_says_fall = fb_label == "negative"
    rf_says_rise = rf_label == "Rise"
    rf_says_fall = rf_label == "Fall"

    conflict = (fb_says_rise and rf_says_fall) or (fb_says_fall and rf_says_rise)

    decision_src = "RF model"

    if conflict:
        if fb_score >= FINBERT_OVERRIDE_THRESHOLD:
            # FinBERT is very confident → trust it over RF
            # e.g. NEGATIVE at 94.3% → FALL regardless of RF
            final_label = "Fall" if fb_says_fall else "Rise"
            final_conf  = int(fb_score * 100)
            final_rise  = fb_score if fb_says_rise else (1 - fb_score)
            final_fall  = (1 - fb_score) if fb_says_rise else fb_score
            decision_src = f"FinBERT override (confidence {int(fb_score*100)}% ≥ {int(FINBERT_OVERRIDE_THRESHOLD*100)}% threshold)"

        elif rf_conf >= RF_CONFIDENCE_THRESHOLD:
            # RF is confident enough → trust it
            final_label = rf_label
            final_conf  = int(rf_conf * 100)
            final_rise  = prob_rise
            final_fall  = prob_fall
            decision_src = f"RF model (confidence {int(rf_conf*100)}% ≥ {int(RF_CONFIDENCE_THRESHOLD*100)}% threshold)"

        else:
            # Both uncertain → weighted blend
            # FinBERT polarity (-1 to +1) blended with RF probability
            blend_score  = (fb_polarity * 0.55) + ((prob_rise - 0.5) * 2 * 0.45)
            final_label  = "Rise" if blend_score >= 0 else "Fall"
            final_conf   = int(min(95, max(50, 50 + abs(blend_score) * 40)))
            final_rise   = min(0.95, max(0.05, 0.5 + blend_score * 0.4))
            final_fall   = round(1 - final_rise, 4)
            decision_src = "Blended (FinBERT + RF, both low confidence)"

    else:
        # No conflict — they agree
        final_label  = rf_label
        final_conf   = int(rf_conf * 100)
        final_rise   = prob_rise
        final_fall   = prob_fall
        decision_src = "RF model (agrees with FinBERT)"

    return _build_result(
        prediction   = final_label,
        confidence   = final_conf,
        prob_rise    = round(final_rise, 4),
        prob_fall    = round(final_fall, 4),
        fb_label     = fb_label,
        fb_score     = fb_score,
        fb_score_pct = fb_score_pct,
        fb_polarity  = fb_polarity,
        model_ready  = True,
        decision_src = decision_src,
        conflict     = conflict,
        stock        = stock,
        features     = list(X.columns),
    )


def _build_result(prediction, confidence, prob_rise, prob_fall,
                  fb_label, fb_score, fb_score_pct, fb_polarity,
                  model_ready, decision_src, stock, features,
                  conflict=False) -> dict:
    """Build the standardised result dict returned to Flask."""

    signal = _interpret_signal(prediction, confidence, fb_label)

    return {
        # ── Core prediction ──
        "prediction":    prediction,
        "confidence":    confidence,
        "prob_rise":     prob_rise,
        "prob_fall":     prob_fall,
        "model_ready":   model_ready,

        # ── FinBERT result ──
        "finbert": {
            "label":     fb_label,
            "score":     fb_score,
            "score_pct": fb_score_pct,
            "polarity":  fb_polarity,
            "emoji":     "🟢" if fb_label == "positive" else
                         ("🔴" if fb_label == "negative" else "🟡"),
        },

        # ── Decision metadata ──
        "decision_source": decision_src,
        "conflict_detected": conflict,

        # ── Signal interpretation ──
        "signal":       signal["text"],
        "signal_color": signal["color"],
        "action":       signal["action"],

        # ── Context ──
        "stock":         stock,
        "features_used": features,
        "n_features":    len(features),
    }


def _interpret_signal(prediction, confidence, fb_label) -> dict:
    """Human-readable signal — aware of sentiment+prediction combination."""

    if prediction == "Rise" and confidence >= 75 and fb_label == "positive":
        return {"text": "Strong buy signal",    "color": "green",
                "action": "Sentiment strongly positive — corroborates upward movement"}
    elif prediction == "Rise" and confidence >= 60:
        return {"text": "Moderate buy signal",  "color": "green",
                "action": "Positive signals detected — consider as supporting evidence"}
    elif prediction == "Fall" and confidence >= 75 and fb_label == "negative":
        return {"text": "Strong sell signal",   "color": "red",
                "action": "Sentiment strongly negative — may indicate downward pressure"}
    elif prediction == "Fall" and confidence >= 60:
        return {"text": "Moderate sell signal", "color": "red",
                "action": "Negative signals detected — monitor closely"}
    elif prediction == "Rise" and fb_label == "negative":
        return {"text": "Conflicting signals",  "color": "amber",
                "action": "Sentiment negative but technical signal positive — high uncertainty, avoid acting"}
    elif prediction == "Fall" and fb_label == "positive":
        return {"text": "Conflicting signals",  "color": "amber",
                "action": "Sentiment positive but technical signal negative — wait for confirmation"}
    else:
        return {"text": "Uncertain signal",     "color": "amber",
                "action": "Insufficient signal strength — do not act on prediction alone"}


# ════════════════════════════════════════════════════
#  BATCH PREDICTION
# ════════════════════════════════════════════════════
def predict_batch(tweets: list, stock: str = "UNKNOWN") -> list:
    results = []
    history = []
    for tweet in tweets:
        result = predict_stock(tweet_text=tweet, stock=stock, history=history.copy())
        history.append(result["finbert"]["polarity"])
        if len(history) > 20: history = history[-20:]
        results.append(result)
    return results


# ════════════════════════════════════════════════════
#  MODEL INFO
# ════════════════════════════════════════════════════
def get_model_info() -> dict:
    metrics = get_metrics()
    rf      = get_model()
    return {
        "model_ready":    rf is not None,
        "cv_accuracy":    metrics.get("cv_accuracy",   0) if metrics else 0,
        "cv_std":         metrics.get("cv_std",        0) if metrics else 0,
        "test_accuracy":  metrics.get("test_accuracy", 0) if metrics else 0,
        "auc_roc":        metrics.get("auc_roc",       0) if metrics else 0,
        "f1_weighted":    metrics.get("f1_weighted",   0) if metrics else 0,
        "precision":      metrics.get("precision",     0) if metrics else 0,
        "recall":         metrics.get("recall",        0) if metrics else 0,
        "tp":             metrics.get("tp", 0) if metrics else 0,
        "tn":             metrics.get("tn", 0) if metrics else 0,
        "fp":             metrics.get("fp", 0) if metrics else 0,
        "fn":             metrics.get("fn", 0) if metrics else 0,
        "total_samples":  metrics.get("total_samples", 0) if metrics else 0,
        "n_features":     metrics.get("n_features",   0) if metrics else 0,
        "feature_importances": metrics.get("feature_importances", {}) if metrics else {},
        "feature_names":  metrics.get("feature_names", []) if metrics else [],
        "finbert_override_threshold": FINBERT_OVERRIDE_THRESHOLD,
        "rf_confidence_threshold":    RF_CONFIDENCE_THRESHOLD,
    }