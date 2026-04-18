import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import transform_input

LABEL_MAP = {0: 'Negative', 1: 'Moderate', 2: 'Positive'}
LABEL_COLORS = {'Negative': '#E24B4A', 'Moderate': '#F9CB42', 'Positive': '#1D9E75'}
LABEL_EMOJIS = {'Negative': '😞', 'Moderate': '😐', 'Positive': '😊'}
LABEL_DESCRIPTIONS = {
    'Negative': 'This restaurant is likely to receive poor customer feedback. Common issues include slow delivery, food quality, or pricing concerns.',
    'Moderate': 'This restaurant is likely to receive average feedback. Customers generally find it acceptable but nothing exceptional.',
    'Positive': 'This restaurant is likely to receive excellent customer feedback. Customers are generally very satisfied.',
}

def load_model():
    """Load trained model from disk."""
    if not os.path.exists('model/analyzer_model.pkl'):
        raise FileNotFoundError("Model not found. Run scripts/train.py first.")
    model = joblib.load('model/analyzer_model.pkl')
    return model

def predict(input_dict: dict) -> dict:
    """
    Predict feedback for a single input.

    Args:
        input_dict: dict with keys:
            name (str), cuisines (str), rest_type (str), location (str),
            listed_in(type) (str), online_order (str: 'Yes'/'No'),
            book_table (str: 'Yes'/'No'), approx_cost(for two people) (int/str), votes (int)

    Returns:
        dict with keys: label (str), label_id (int), confidence (float),
                        probabilities (dict), color (str), emoji (str), description (str)
    """
    model = load_model()
    X = transform_input(input_dict)
    pred_id = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    label = LABEL_MAP[pred_id]
    confidence = float(proba[pred_id]) * 100

    return {
        'label': label,
        'label_id': int(pred_id),
        'confidence': round(confidence, 2),
        'probabilities': {
            'Negative': round(float(proba[0]) * 100, 2),
            'Moderate': round(float(proba[1]) * 100, 2),
            'Positive': round(float(proba[2]) * 100, 2),
        },
        'color': LABEL_COLORS[label],
        'emoji': LABEL_EMOJIS[label],
        'description': LABEL_DESCRIPTIONS[label],
    }
