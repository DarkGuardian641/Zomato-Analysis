import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize

# Add parent dir to path so we can import preprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import load_and_prepare

LABEL_NAMES = ['Negative', 'Moderate', 'Positive']
CSV_PATH = 'data/processed/clean_zomato.csv'

def train():
    os.makedirs('model', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # ---- Step 1: Load & preprocess ----
    print("=" * 60)
    print("STEP 1: Loading and preprocessing data")
    print("=" * 60)
    X, y, df = load_and_prepare(CSV_PATH)

    # ---- Step 2: Train/test split ----
    print("\n" + "=" * 60)
    print("STEP 2: Splitting data (80% train, 20% test, stratified)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape[0]:,}")
    print(f"Test size:  {X_test.shape[0]:,}")

    # ---- Step 3: Train RandomForest ----
    print("\n" + "=" * 60)
    print("STEP 3: Training RandomForestClassifier")
    print("=" * 60)
    print("Parameters:")
    print("  n_estimators = 300")
    print("  class_weight = 'balanced'  (handles class imbalance)")
    print("  n_jobs = -1  (use all CPU cores)")
    print("  min_samples_split = 5")
    print("  min_samples_leaf = 2")
    print("\nTraining... (may take 2–5 minutes)")

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # ---- Step 4: Evaluate ----
    print("\n" + "=" * 60)
    print("STEP 4: Evaluating on test set")
    print("=" * 60)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # ---- Step 5: Cross-validation ----
    print("=" * 60)
    print("STEP 5: 5-Fold Stratified Cross-Validation")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"CV Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ---- Step 6: Save model ----
    print("\n" + "=" * 60)
    print("STEP 6: Saving model")
    print("=" * 60)
    joblib.dump(model, 'model/analyzer_model.pkl')
    print("Saved: model/analyzer_model.pkl")

    # ---- Step 7: Confusion Matrix ----
    print("\n" + "=" * 60)
    print("STEP 7: Generating evaluation plots")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Zomato Analyzer Model — Evaluation', fontsize=14, fontweight='bold')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=axes[0], cmap='Oranges', colorbar=False)
    axes[0].set_title('Confusion Matrix', fontweight='bold')

    # ROC curves (one-vs-rest)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    colors_roc = ['#E24B4A', '#F9CB42', '#1D9E75']
    for i, (label, color) in enumerate(zip(LABEL_NAMES, colors_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
        axes[1].plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})', color=color, lw=2)
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curves (One-vs-Rest)', fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/evaluation.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/evaluation.png")
    plt.close()

    # ---- Step 8: Class distribution plot ----
    label_map = {0: 'Negative', 1: 'Moderate', 2: 'Positive'}
    df['feedback'] = df['label'].map(label_map)
    counts = df['feedback'].value_counts()[['Negative', 'Moderate', 'Positive']]
    colors = ['#E24B4A', '#F9CB42', '#1D9E75']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(counts.index, counts.values, color=colors, alpha=0.9, width=0.5, edgecolor='white')
    axes[0].set_title('Feedback Class Distribution', fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 100, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=10)

    axes[1].pie(counts.values, labels=counts.index, colors=colors, autopct='%1.1f%%',
                startangle=90, pctdistance=0.75, wedgeprops=dict(linewidth=1, edgecolor='white'))
    axes[1].set_title('Feedback Proportions', fontweight='bold')

    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/class_distribution.png")
    plt.close()

    # ---- Step 9: Feature importance ----
    tfidf = joblib.load('model/tfidf_vectorizer.pkl')
    label_encoders = joblib.load('model/label_encoders.pkl')
    feature_names = (
        list(tfidf.get_feature_names_out()) +
        [col + '_enc' for col in ['online_order', 'book_table', 'rest_type', 'cuisines', 'location', 'listed_in(type)']] +
        ['cost_num', 'votes']
    )
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    top_names = [feature_names[i] if i < len(feature_names) else f'feat_{i}' for i in top_idx]
    top_vals = importances[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_names[::-1], top_vals[::-1], color='#E23744', alpha=0.85)
    plt.title('Top 20 Feature Importances', fontweight='bold')
    plt.xlabel('Importance score')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
    print("Saved: outputs/feature_importance.png")
    plt.close()

    # ---- Step 10: Save metrics summary ----
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    metrics_df = pd.DataFrame({
        'Class': LABEL_NAMES,
        'Precision': precision.round(4),
        'Recall': recall.round(4),
        'F1-Score': f1.round(4),
        'Support': support,
    })
    metrics_df.to_csv('outputs/metrics.csv', index=False)
    print("Saved: outputs/metrics.csv")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"CV Mean Accuracy    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nModel artifacts:")
    print("  model/analyzer_model.pkl")
    print("  model/tfidf_vectorizer.pkl")
    print("  model/label_encoders.pkl")
    print("\nOutput plots:")
    print("  outputs/evaluation.png")
    print("  outputs/class_distribution.png")
    print("  outputs/feature_importance.png")
    print("  outputs/metrics.csv")

if __name__ == '__main__':
    train()
