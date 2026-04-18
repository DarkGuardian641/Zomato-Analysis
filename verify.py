import os, joblib

checks = [
    ('data/processed/clean_zomato.csv', 'Dataset'),
    ('scripts/preprocess.py', 'Preprocess script'),
    ('scripts/train.py', 'Train script'),
    ('scripts/predict.py', 'Predict script'),
    ('main.py', 'Streamlit app'),
    ('model/analyzer_model.pkl', 'Trained model'),
    ('model/tfidf_vectorizer.pkl', 'TF-IDF vectorizer'),
    ('model/label_encoders.pkl', 'Label encoders'),
    ('outputs/evaluation.png', 'Evaluation plot'),
    ('outputs/metrics.csv', 'Metrics CSV'),
]

print("=" * 50)
print("VERIFICATION REPORT")
print("=" * 50)
all_ok = True
for path, name in checks:
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {name} ({path})")
    if not exists:
        all_ok = False

print("=" * 50)
if all_ok:
    print("All files present. Run: streamlit run main.py")
else:
    print("Some files missing. Re-run the training step.")

# Quick inference test
if os.path.exists('model/analyzer_model.pkl'):
    import sys
    sys.path.insert(0, 'scripts')
    from predict import predict
    test_input = {
        'name': 'Test Biryani House',
        'cuisines': 'North Indian, Biryani',
        'rest_type': 'Quick Bites',
        'location': 'Koramangala 5th Block',
        'listed_in(type)': 'Delivery',
        'online_order': 'Yes',
        'book_table': 'No',
        'approx_cost(for two people)': 300,
        'votes': 50,
    }
    result = predict(test_input)
    print(f"\nSample prediction:")
    print(f"  Input: {test_input['name']} — {test_input['cuisines']}")
    print(f"  Prediction: {result['emoji']} {result['label']} ({result['confidence']:.1f}% confidence)")
    print(f"  Probabilities: {result['probabilities']}")
