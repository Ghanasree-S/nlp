"""
KEYPHRASE EXTRACTOR - Gradient Boosting Classifier (MULTI-DATASET)
Model 2 of 4
- Uses ALL available keyphrase datasets
- Better feature engineering with TF-IDF features
- Optimized sampling and matching
- Tuned hyperparameters for high F1
"""

import os
import numpy as np
import pickle
import json
import re
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import spacy
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("KEYPHRASE EXTRACTOR - GRADIENT BOOSTING (MULTI-DATASET)")
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

# Load SpaCy model
print("Loading SpaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model loaded successfully")
except:
    print("⚠️ SpaCy model not found. Installing...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model installed and loaded")
print()

def normalize_text(text):
    """Normalize text for comparison"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def load_dataset_from_folder(dataset_path, max_docs=200):
    """Load documents and keyphrases from a dataset folder"""
    docs = []
    keyphrases = []
    
    docsutf8_dir = dataset_path / "docsutf8"
    keys_dir = dataset_path / "keys"
    
    if not docsutf8_dir.exists() or not keys_dir.exists():
        return [], []
    
    txt_files = sorted(list(docsutf8_dir.glob("*.txt")))[:max_docs]
    
    for txt_file in txt_files:
        file_id = txt_file.stem
        key_file = keys_dir / f"{file_id}.key"
        
        if key_file.exists():
            try:
                with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                    doc_text = f.read().strip()
                with open(key_file, 'r', encoding='utf-8', errors='ignore') as f:
                    kps = [line.strip() for line in f.readlines() if line.strip()]
                
                if doc_text and kps and len(doc_text) > 50:
                    docs.append(doc_text)
                    keyphrases.append(kps)
            except:
                continue
    
    return docs, keyphrases

def load_all_keyphrase_data():
    """Load data from ALL available keyphrase datasets"""
    print("📚 Loading Keyphrase data from ALL datasets...")
    
    all_docs = []
    all_keyphrases = []
    
    datasets_dir = DATASET_DIR / "datasets"
    
    # Use datasets with cleaner keyphrase annotations first
    datasets_config = [
        ("Inspec", 500),        # High quality, short abstracts
        ("SemEval2010", 200),
        ("SemEval2017", 200),
        ("Krapivin2009", 200),
        ("500N-KPCrowd-v1.1", 200),
        ("PubMed", 150),
        ("Nguyen2007", 150),
        ("kdd", 150),
        ("www", 150),
        ("fao780", 100),
    ]
    
    for dataset_name, max_docs in datasets_config:
        dataset_path = datasets_dir / dataset_name
        if dataset_path.exists():
            docs, kps = load_dataset_from_folder(dataset_path, max_docs)
            if docs:
                all_docs.extend(docs)
                all_keyphrases.extend(kps)
                print(f"  ✅ {dataset_name}: {len(docs)} documents")
    
    print(f"  📊 Total documents loaded: {len(all_docs)}")
    return all_docs, all_keyphrases

def extract_keyphrase_candidates(doc, doc_text):
    """Extract keyphrase candidates using SpaCy - comprehensive extraction"""
    candidates = set()
    
    # Extract named entities
    for ent in doc.ents:
        if ent.text.strip() and len(ent.text) > 2:
            candidates.add(normalize_text(ent.text))
    
    # Extract noun chunks (most important for keyphrases)
    for chunk in doc.noun_chunks:
        tokens = [t for t in chunk if t.pos_ not in ['DET', 'PRON']]
        if tokens:
            phrase = ' '.join([t.text for t in tokens])
            if len(phrase) > 2 and len(phrase.split()) <= 4:
                candidates.add(normalize_text(phrase))
            # Also add the head noun alone
            head = chunk.root
            if head.pos_ in ['NOUN', 'PROPN'] and len(head.text) > 2:
                candidates.add(normalize_text(head.lemma_))
    
    # Extract nouns and compound nouns
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.pos_ in ['NOUN', 'PROPN']:
            phrase_tokens = [token]
            
            j = i + 1
            while j < len(doc) and doc[j].pos_ in ['NOUN', 'PROPN']:
                phrase_tokens.append(doc[j])
                j += 1
            
            k = i - 1
            adj_tokens = []
            while k >= 0 and doc[k].pos_ == 'ADJ':
                adj_tokens.insert(0, doc[k])
                k -= 1
            
            all_tokens = adj_tokens + phrase_tokens
            phrase = ' '.join([t.text for t in all_tokens])
            if len(phrase) > 2:
                candidates.add(normalize_text(phrase))
            
            # Add individual nouns too
            for t in phrase_tokens:
                if len(t.text) > 3 and not t.is_stop:
                    candidates.add(normalize_text(t.lemma_))
            
            i = j
        else:
            i += 1
    
    return list(candidates)

def is_keyphrase_match(candidate, keyphrases_normalized):
    """Check if candidate matches any keyphrase - balanced matching"""
    candidate_norm = normalize_text(candidate)
    if len(candidate_norm) < 2:
        return False
    
    for kp in keyphrases_normalized:
        # Exact match
        if candidate_norm == kp:
            return True
        # Substring match (both directions)
        if len(candidate_norm) > 3 and len(kp) > 3:
            if candidate_norm in kp or kp in candidate_norm:
                return True
        # Word overlap match
        cand_words = set(candidate_norm.split())
        kp_words = set(kp.split())
        if len(cand_words) > 0 and len(kp_words) > 0:
            overlap = len(cand_words & kp_words) / max(len(cand_words), len(kp_words))
            if overlap >= 0.6:
                return True
        # Stem-level matching for single words
        if len(candidate_norm.split()) == 1 and len(kp.split()) == 1:
            min_len = min(len(candidate_norm), len(kp))
            prefix_len = min(min_len - 1, 6)
            if prefix_len >= 4 and candidate_norm[:prefix_len] == kp[:prefix_len]:
                return True
    
    return False

def create_keyphrase_features(doc, candidate, doc_text, tfidf_score=0.0):
    """Create features for a keyphrase candidate"""
    candidate_lower = candidate.lower()
    doc_text_lower = doc_text.lower()
    doc_len = max(len(doc_text), 1)
    doc_words = max(len(doc_text.split()), 1)
    
    # Basic features
    word_count = len(candidate.split())
    char_count = len(candidate)
    
    # Position features
    first_pos = doc_text_lower.find(candidate_lower)
    position = first_pos / doc_len if first_pos != -1 else 1.0
    in_first_10 = 1 if position < 0.1 else 0
    in_first_20 = 1 if position < 0.2 else 0
    in_first_50 = 1 if position < 0.5 else 0
    
    # Frequency features
    frequency = doc_text_lower.count(candidate_lower)
    tf = frequency / doc_words
    log_freq = np.log1p(frequency)
    
    # Sentence features
    sentences = doc_text.split('.')
    first_sentence = sentences[0].lower() if sentences else ''
    in_first_sentence = 1 if candidate_lower in first_sentence else 0
    
    # Title features
    first_line = doc_text.split('\n')[0].lower() if '\n' in doc_text else doc_text[:200].lower()
    in_title = 1 if candidate_lower in first_line else 0
    
    # POS features
    cand_doc = nlp(candidate)
    has_noun = 1 if any(t.pos_ in ['NOUN', 'PROPN'] for t in cand_doc) else 0
    has_adj = 1 if any(t.pos_ == 'ADJ' for t in cand_doc) else 0
    all_noun = 1 if all(t.pos_ in ['NOUN', 'PROPN', 'ADJ'] for t in cand_doc if not t.is_punct) else 0
    
    # Capitalization
    is_capitalized = 1 if candidate and candidate[0].isupper() else 0
    has_numbers = 1 if any(c.isdigit() for c in candidate) else 0
    
    # Length-based features
    avg_word_len = np.mean([len(w) for w in candidate.split()]) if candidate.split() else 0
    
    # Spread: how distributed is the candidate across the document
    positions = []
    start = 0
    while True:
        pos = doc_text_lower.find(candidate_lower, start)
        if pos == -1:
            break
        positions.append(pos / doc_len)
        start = pos + 1
    spread = np.std(positions) if len(positions) > 1 else 0.0
    
    features = [
        word_count, char_count, position, 
        in_first_10, in_first_20, in_first_50,
        frequency, tf * 100, log_freq,
        in_first_sentence, in_title,
        has_noun, has_adj, all_noun,
        is_capitalized, has_numbers,
        avg_word_len, spread,
        tfidf_score,
    ]
    
    return features

# Load data from all datasets
X_docs, y_keyphrases = load_all_keyphrase_data()

if len(X_docs) == 0:
    print("❌ No keyphrase data found!")
    exit(1)

# Build TF-IDF model on all documents for scoring
print()
print("  📊 Building TF-IDF model...")
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(X_docs)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"  ✅ TF-IDF vocabulary size: {len(tfidf_feature_names)}")

def get_tfidf_score(candidate, doc_idx):
    """Get TF-IDF score for a candidate in a document"""
    candidate_lower = candidate.lower()
    if candidate_lower in tfidf_feature_names:
        feat_idx = list(tfidf_feature_names).index(candidate_lower)
        return tfidf_matrix[doc_idx, feat_idx]
    return 0.0

# Prepare training data
print()
print("  🔧 Preparing keyphrase training data...")
X_features = []
y_labels = []
doc_indices = []

num_docs = len(X_docs)
print(f"  📊 Processing {num_docs} documents...")

for i in range(num_docs):
    if i % 200 == 0:
        print(f"    Processing document {i+1}/{num_docs}...")
    
    doc_text = X_docs[i]
    true_keyphrases = y_keyphrases[i]
    true_keyphrases_normalized = [normalize_text(kp) for kp in true_keyphrases]
    
    doc = nlp(doc_text[:5000])  # Limit for speed
    candidates = extract_keyphrase_candidates(doc, doc_text)
    
    positive_samples = []
    negative_samples = []
    
    for candidate in candidates:
        if not candidate or len(candidate) < 2:
            continue
        tfidf_score = get_tfidf_score(candidate, i)
        features = create_keyphrase_features(doc, candidate, doc_text, tfidf_score)
        is_kp = is_keyphrase_match(candidate, true_keyphrases_normalized)
        
        if is_kp:
            positive_samples.append((features, 1))
        else:
            negative_samples.append((features, 0))
    
    # Balance: 1:1 ratio for best F1
    max_negatives = max(len(positive_samples), 2)
    if len(negative_samples) > max_negatives:
        np.random.seed(RANDOM_STATE + i)
        indices = np.random.choice(len(negative_samples), max_negatives, replace=False)
        negative_samples = [negative_samples[j] for j in indices]
    
    for features, label in positive_samples + negative_samples:
        X_features.append(features)
        y_labels.append(label)
        doc_indices.append(i)

X_features = np.array(X_features)
y_labels = np.array(y_labels)
doc_indices = np.array(doc_indices)

print(f"  📊 Total training samples: {len(X_features)}")
print(f"  📊 Positive samples: {sum(y_labels)} ({sum(y_labels)/len(y_labels)*100:.1f}%)")
print(f"  📊 Negative samples: {len(y_labels) - sum(y_labels)} ({(len(y_labels)-sum(y_labels))/len(y_labels)*100:.1f}%)")
print()

# K-Fold Cross-Validation (document-level splits)
print(f"🔄 Starting {K_FOLDS}-Fold Cross-Validation...")
print()

keyphrase_results = {
    'fold_results': [],
    'avg_metrics': {}
}

unique_docs = np.unique(doc_indices)
kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

for fold_idx, (train_doc_idx, test_doc_idx) in enumerate(kfold.split(unique_docs), 1):
    print(f"  Fold {fold_idx}/{K_FOLDS}")
    print("  " + "-" * 40)
    
    train_docs = unique_docs[train_doc_idx]
    test_docs = unique_docs[test_doc_idx]
    
    train_mask = np.isin(doc_indices, train_docs)
    test_mask = np.isin(doc_indices, test_docs)
    
    X_train, X_test = X_features[train_mask], X_features[test_mask]
    y_train, y_test = y_labels[train_mask], y_labels[test_mask]
    
    clf = GradientBoostingClassifier(
        n_estimators=300, 
        learning_rate=0.05,
        max_depth=7, 
        min_samples_leaf=2,
        min_samples_split=4,
        subsample=0.8,
        max_features='sqrt',
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print()
    
    keyphrase_results['fold_results'].append({
        'fold': fold_idx,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Calculate average metrics
avg_accuracy = np.mean([r['accuracy'] for r in keyphrase_results['fold_results']])
avg_precision = np.mean([r['precision'] for r in keyphrase_results['fold_results']])
avg_recall = np.mean([r['recall'] for r in keyphrase_results['fold_results']])
avg_f1 = np.mean([r['f1_score'] for r in keyphrase_results['fold_results']])

keyphrase_results['avg_metrics'] = {
    'accuracy': avg_accuracy,
    'precision': avg_precision,
    'recall': avg_recall,
    'f1_score': avg_f1
}

print("  " + "=" * 40)
print(f"  📊 AVERAGE METRICS (K-Fold CV)")
print("  " + "=" * 40)
print(f"    Accuracy:  {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
print(f"    Precision: {avg_precision:.4f} ({avg_precision*100:.2f}%)")
print(f"    Recall:    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
print(f"    F1 Score:  {avg_f1:.4f} ({avg_f1*100:.2f}%)")
print()

# Train final model
print("  🎯 Training final model on full dataset...")
clf_final = GradientBoostingClassifier(
    n_estimators=300, 
    learning_rate=0.05,
    max_depth=7,
    min_samples_leaf=2,
    min_samples_split=4,
    subsample=0.8,
    max_features='sqrt',
    random_state=RANDOM_STATE
)
clf_final.fit(X_features, y_labels)

# Save model
model_path = OUTPUT_DIR / "keyphrase_extractor_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': clf_final,
        'feature_names': ['word_count', 'char_count', 'position', 
                         'in_first_10', 'in_first_20', 'in_first_50',
                         'frequency', 'tf', 'log_freq',
                         'in_first_sentence', 'in_title', 'has_noun',
                         'has_adj', 'all_noun', 'is_capitalized', 'has_numbers',
                         'avg_word_len', 'spread', 'tfidf_score']
    }, f)
print(f"  ✅ Model saved to {model_path}")

# Save results
results_path = RESULTS_DIR / "keyphrase_extractor_results.json"
with open(results_path, 'w') as f:
    json.dump(keyphrase_results, f, indent=2)
print(f"  ✅ Results saved to {results_path}")

print()
print("=" * 80)
print("KEYPHRASE EXTRACTOR TRAINING COMPLETE!")
print("=" * 80)
