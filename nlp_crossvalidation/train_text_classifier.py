"""
TEXT CLASSIFIER - Hybrid Random Forest Ensemble
Model 1 of 4
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TEXT CLASSIFIER - HYBRID RANDOM FOREST ENSEMBLE")
print("=" * 80)
print()

# Configuration
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("models")
RESULTS_DIR = Path("results")
K_FOLDS = 5
RANDOM_STATE = 42

# Create output directories
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

print(f"📁 Dataset Directory: {DATASET_DIR}")
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"🔢 K-Folds: {K_FOLDS}")
print()

def load_text_classifier_data():
    """Load fairy tales (narrative) and news/Wikipedia (informational)"""
    print("📚 Loading Text Classifier data...")
    
    texts = []
    labels = []
    
    # Load fairy tales (Narrative - label 0)
    fairy_tales_path = DATASET_DIR / "cleaned_merged_fairy_tales_without_eos.txt"
    if fairy_tales_path.exists():
        with open(fairy_tales_path, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = content.split('\n\n')
            for chunk in chunks[:1000]:
                if len(chunk.strip()) > 100:
                    texts.append(chunk.strip()[:500])
                    labels.append(0)
        print(f"  ✅ Loaded {len([l for l in labels if l == 0])} narrative texts")
    else:
        print(f"  ⚠️ Fairy tales file not found: {fairy_tales_path}")
    
    # Load BBC News (Informational - label 1)
    news_path = DATASET_DIR / "bbc-news-data.csv"
    if news_path.exists():
        df = pd.read_csv(news_path, sep='\t', on_bad_lines='skip')
        for idx, row in df.head(1000).iterrows():
            if pd.notna(row['content']) and len(str(row['content'])) > 100:
                texts.append(str(row['content'])[:500])
                labels.append(1)
        print(f"  ✅ Loaded {len([l for l in labels if l == 1])} informational texts")
    else:
        print(f"  ⚠️ BBC News file not found: {news_path}")
    
    print(f"  📊 Total samples: {len(texts)}")
    return np.array(texts), np.array(labels)

# Load data
X_text, y_text = load_text_classifier_data()

if len(X_text) < 10:
    print("❌ Insufficient data for training!")
    exit(1)

# Initialize K-Fold
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

text_classifier_results = {
    'fold_results': [],
    'fold_reports': [],
    'avg_metrics': {}
}

print(f"\n🔄 Starting {K_FOLDS}-Fold Cross-Validation...")
print()

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_text), 1):
    print(f"  Fold {fold_idx}/{K_FOLDS}")
    print("  " + "-" * 40)
    
    X_train, X_test = X_text[train_idx], X_text[test_idx]
    y_train, y_test = y_text[train_idx], y_text[test_idx]
    
    # Feature extraction with TF-IDF
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), 
                                  stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Hybrid Random Forest
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, 
                                   random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print()
    
    text_classifier_results['fold_results'].append({
        'fold': fold_idx,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    
    report = classification_report(y_test, y_pred, 
                                   target_names=['Narrative', 'Informational'],
                                   output_dict=True, zero_division=0)
    text_classifier_results['fold_reports'].append(report)

# Calculate average metrics
avg_accuracy = np.mean([r['accuracy'] for r in text_classifier_results['fold_results']])
avg_precision = np.mean([r['precision'] for r in text_classifier_results['fold_results']])
avg_recall = np.mean([r['recall'] for r in text_classifier_results['fold_results']])
avg_f1 = np.mean([r['f1_score'] for r in text_classifier_results['fold_results']])

text_classifier_results['avg_metrics'] = {
    'accuracy': avg_accuracy,
    'precision': avg_precision,
    'recall': avg_recall,
    'f1_score': avg_f1
}

print("  " + "=" * 40)
print(f"  📊 AVERAGE METRICS (K-Fold CV)")
print("  " + "=" * 40)
print(f"    Accuracy:  {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
print(f"    Precision: {avg_precision:.4f}")
print(f"    Recall:    {avg_recall:.4f}")
print(f"    F1 Score:  {avg_f1:.4f}")
print()

# Train final model on full data
print("  🎯 Training final model on full dataset...")
vectorizer_final = TfidfVectorizer(max_features=500, ngram_range=(1, 2), 
                                    stop_words='english')
X_full_tfidf = vectorizer_final.fit_transform(X_text)
clf_final = RandomForestClassifier(n_estimators=100, max_depth=20, 
                                    random_state=RANDOM_STATE, n_jobs=-1)
clf_final.fit(X_full_tfidf, y_text)

# Save model
model_path = OUTPUT_DIR / "text_classifier_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': clf_final,
        'vectorizer': vectorizer_final,
        'label_names': ['Narrative', 'Informational']
    }, f)
print(f"  ✅ Model saved to {model_path}")

# Save results
results_path = RESULTS_DIR / "text_classifier_results.json"
with open(results_path, 'w') as f:
    json.dump(text_classifier_results, f, indent=2)
print(f"  ✅ Results saved to {results_path}")

print()
print("=" * 80)
print("TEXT CLASSIFIER TRAINING COMPLETE!")
print("=" * 80)
