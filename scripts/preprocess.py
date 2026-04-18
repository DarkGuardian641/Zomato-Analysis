import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import joblib
import os

STRUCT_COLS = ['online_order', 'book_table', 'rest_type', 'cuisines', 'location', 'listed_in(type)']

def assign_feedback(rate):
    """
    Maps numeric rating to feedback label.
    < 3.5  → Negative (0)
    3.5–3.9 → Moderate (1)
    >= 4.0 → Positive (2)
    """
    if rate < 3.5:
        return 0
    elif rate < 4.0:
        return 1
    else:
        return 2

def clean_cost(val):
    """Parse cost string like '1,000' into float."""
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return 500.0

def build_text(row):
    """
    Construct a single text string from structured fields.
    This is what TF-IDF will be trained on.
    Used for both training and inference.
    """
    parts = [
        str(row.get('name', '')),
        str(row.get('cuisines', '')),
        str(row.get('rest_type', '')),
        str(row.get('location', '')),
        str(row.get('listed_in(type)', '')),
        'online order available' if str(row.get('online_order', '')).strip() == 'Yes' else 'no online order',
        'table booking available' if str(row.get('book_table', '')).strip() == 'Yes' else 'no table booking',
    ]
    return ' '.join(parts).lower().strip()

def load_and_prepare(csv_path: str):
    """
    Full pipeline: load CSV → clean → label → return X (combined sparse matrix), y (labels), and df.
    Also fits and saves the TF-IDF vectorizer and label encoders.
    """
    os.makedirs('model', exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Clean cost column ---
    df['cost_num'] = df['approx_cost(for two people)'].apply(clean_cost)

    # --- Fill nulls in text cols ---
    for col in STRUCT_COLS:
        df[col] = df[col].fillna('Unknown')

    # --- Create feedback label ---
    df['label'] = df['rate'].apply(assign_feedback)
    label_counts = df['label'].value_counts().sort_index()
    print(f"Label distribution:")
    print(f"  Negative (0): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  Moderate (1): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"  Positive (2): {label_counts.get(2, 0):,} ({label_counts.get(2, 0)/len(df)*100:.1f}%)")

    # --- Build synthetic text feature ---
    df['text'] = df.apply(build_text, axis=1)

    # --- Fit TF-IDF ---
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
    )
    X_text = tfidf.fit_transform(df['text'])
    joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')
    print(f"TF-IDF matrix: {X_text.shape}")

    # --- Fit Label Encoders for structured cols ---
    label_encoders = {}
    struct_encoded = []
    for col in STRUCT_COLS:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col])
        label_encoders[col] = le
        struct_encoded.append(encoded)

    joblib.dump(label_encoders, 'model/label_encoders.pkl')

    # --- Stack structured features ---
    X_struct = np.column_stack(struct_encoded + [df['cost_num'].values, df['votes'].values])
    X_struct_sparse = sp.csr_matrix(X_struct.astype(float))

    # --- Combine TF-IDF + structured ---
    X_combined = sp.hstack([X_text, X_struct_sparse])
    y = df['label'].values

    print(f"Final feature matrix: {X_combined.shape}")
    return X_combined, y, df

def transform_input(input_dict: dict) -> sp.csr_matrix:
    """
    Transform a single user input dict into the same feature vector used during training.
    input_dict keys: name, cuisines, rest_type, location, listed_in(type),
                     online_order, book_table, approx_cost(for two people), votes
    """
    tfidf = joblib.load('model/tfidf_vectorizer.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')

    row = input_dict.copy()
    row['cost_num'] = clean_cost(row.get('approx_cost(for two people)', 500))

    # Build text
    text = build_text(row)
    X_text = tfidf.transform([text])

    # Encode structured cols
    struct_vals = []
    for col in STRUCT_COLS:
        le = label_encoders[col]
        val = str(row.get(col, 'Unknown')).strip()
        if val in le.classes_:
            encoded = le.transform([val])[0]
        else:
            # Unseen category: use most frequent class (index 0 after sorted encoding)
            encoded = 0
        struct_vals.append(encoded)

    struct_vals.append(float(row.get('cost_num', 500)))
    struct_vals.append(float(row.get('votes', 0)))

    X_struct = sp.csr_matrix(np.array(struct_vals, dtype=float).reshape(1, -1))
    X_combined = sp.hstack([X_text, X_struct])
    return X_combined
