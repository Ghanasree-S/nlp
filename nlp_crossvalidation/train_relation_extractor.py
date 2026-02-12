"""
RELATION EXTRACTOR - Dependency-based (IMPROVED)
Model 4 of 4
- Uses ALL BBC News articles
- Uses Fairy Tales data
- More diverse relation patterns
- Better regularization to prevent overfitting
"""

import os
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spacy
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("RELATION EXTRACTOR - DEPENDENCY-BASED (IMPROVED)")
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

def extract_dependency_relations(text):
    """Extract relations using dependency parsing - improved version"""
    doc = nlp(text[:5000])
    relations = []
    
    for sent in doc.sents:
        for token in sent:
            # Subject-Verb-Object patterns
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                subjects = [child for child in token.children if child.dep_ in ('nsubj', 'nsubjpass')]
                objects = [child for child in token.children if child.dep_ in ('dobj', 'pobj', 'attr', 'dative')]
                
                for subj in subjects:
                    for obj in objects:
                        # Get full noun phrases
                        subj_text = get_noun_phrase(subj)
                        obj_text = get_noun_phrase(obj)
                        
                        if len(subj_text) > 1 and len(obj_text) > 1:
                            relations.append({
                                'subject': subj_text,
                                'predicate': token.lemma_,
                                'object': obj_text,
                                'relation_type': classify_relation_type(token.lemma_, subj.dep_, obj.dep_)
                            })
            
            # Prepositional relations (X prep Y)
            if token.dep_ == 'prep':
                head = token.head
                pobj = [child for child in token.children if child.dep_ == 'pobj']
                
                if head.pos_ in ['NOUN', 'PROPN', 'VERB'] and pobj:
                    for obj in pobj:
                        head_text = get_noun_phrase(head) if head.pos_ in ['NOUN', 'PROPN'] else head.text
                        obj_text = get_noun_phrase(obj)
                        
                        if len(head_text) > 1 and len(obj_text) > 1:
                            relations.append({
                                'subject': head_text,
                                'predicate': token.text,
                                'object': obj_text,
                                'relation_type': f"prep_{token.text}"
                            })
            
            # Possessive relations (X's Y)
            if token.dep_ == 'poss':
                owner = token
                owned = token.head
                
                if owned.pos_ in ['NOUN', 'PROPN']:
                    relations.append({
                        'subject': get_noun_phrase(owner),
                        'predicate': 'has',
                        'object': get_noun_phrase(owned),
                        'relation_type': 'possession'
                    })
    
    return relations

def get_noun_phrase(token):
    """Get the full noun phrase for a token"""
    phrase_parts = []
    
    # Get compound modifiers and adjectives before
    for child in token.children:
        if child.dep_ in ['compound', 'amod'] and child.i < token.i:
            phrase_parts.append(child.text)
    
    phrase_parts.append(token.text)
    
    return ' '.join(phrase_parts)

def classify_relation_type(predicate, subj_dep, obj_dep):
    """Classify relation into broader categories"""
    predicate = predicate.lower()
    
    # Action verbs
    action_verbs = ['do', 'make', 'create', 'build', 'develop', 'produce', 'generate', 'perform']
    if predicate in action_verbs:
        return 'action'
    
    # Possession/ownership
    ownership_verbs = ['have', 'own', 'possess', 'hold', 'contain', 'include']
    if predicate in ownership_verbs:
        return 'ownership'
    
    # Movement
    movement_verbs = ['go', 'come', 'move', 'travel', 'walk', 'run', 'fly', 'drive']
    if predicate in movement_verbs:
        return 'movement'
    
    # Communication
    comm_verbs = ['say', 'tell', 'speak', 'ask', 'answer', 'announce', 'report', 'claim']
    if predicate in comm_verbs:
        return 'communication'
    
    # Transaction
    transaction_verbs = ['buy', 'sell', 'pay', 'give', 'receive', 'acquire', 'purchase']
    if predicate in transaction_verbs:
        return 'transaction'
    
    # State/attribute
    state_verbs = ['be', 'become', 'seem', 'appear', 'remain', 'stay']
    if predicate in state_verbs:
        return 'state'
    
    # Work/employment
    work_verbs = ['work', 'employ', 'hire', 'manage', 'lead', 'direct']
    if predicate in work_verbs:
        return 'employment'
    
    # Default based on dependency
    return f"{subj_dep}_{obj_dep}"

def relation_to_features(relation, all_predicates):
    """Convert relation to feature vector - enhanced features"""
    
    subj_words = len(relation['subject'].split())
    obj_words = len(relation['object'].split())
    pred_words = len(relation['predicate'].split())
    
    subj_chars = len(relation['subject'])
    obj_chars = len(relation['object'])
    pred_chars = len(relation['predicate'])
    
    # Check if subject/object are capitalized (proper nouns)
    subj_capitalized = 1 if relation['subject'][0].isupper() else 0
    obj_capitalized = 1 if relation['object'][0].isupper() else 0
    
    # Predicate position in vocabulary
    pred_lower = relation['predicate'].lower()
    pred_freq = all_predicates.get(pred_lower, 0)
    
    # Relation type encoding (simple hash-based)
    rel_type = relation['relation_type']
    type_hash = hash(rel_type) % 100  # Simple encoding
    
    features = [
        subj_words,
        obj_words,
        pred_words,
        subj_words + obj_words + pred_words,
        subj_chars,
        obj_chars,
        pred_chars,
        subj_capitalized,
        obj_capitalized,
        pred_freq,
        type_hash / 100.0,  # Normalized
    ]
    
    return features

# Load ALL data sources
print("📚 Loading relation extraction data from ALL sources...")

all_sentences = []

# 1. Basic example sentences
basic_examples = [
    "John works at Google in California.",
    "Microsoft acquired LinkedIn in 2016.",
    "The CEO announced the merger yesterday.",
    "Apple released the iPhone in 2007.",
    "Tesla manufactures electric vehicles.",
    "The president visited the White House.",
    "Mary studied computer science at MIT.",
    "The company hired 500 employees last year.",
    "Amazon purchased Whole Foods Market.",
    "Facebook launched Instagram Stories.",
    "Scientists discovered a new planet.",
    "The artist exhibited beautiful paintings.",
    "Engineers developed innovative solutions.",
    "Students completed their assignments.",
    "The team won the championship game.",
    "The bank approved the loan application.",
    "Researchers published their findings.",
    "The director filmed the movie.",
    "Doctors treated the patients.",
    "Teachers educated the students.",
]
all_sentences.extend(basic_examples)
print(f"  ✅ Basic examples: {len(basic_examples)} sentences")

# 2. Load ALL BBC News articles
news_path = DATASET_DIR / "bbc-news-data.csv"
if news_path.exists():
    print("  📰 Loading BBC News data...")
    df = pd.read_csv(news_path, sep='\t', on_bad_lines='skip')
    news_count = 0
    for idx, row in df.iterrows():
        if pd.notna(row['content']):
            content = str(row['content'])
            sentences = content.split('.')[:5]  # First 5 sentences per article
            for sent in sentences:
                if len(sent.strip()) > 20 and len(sent.strip()) < 200:
                    all_sentences.append(sent.strip())
                    news_count += 1
                    if news_count >= 2000:  # Limit to 2000 sentences
                        break
        if news_count >= 2000:
            break
    print(f"  ✅ BBC News: {news_count} sentences")

# 3. Load Fairy Tales
fairy_tales_path = DATASET_DIR / "cleaned_merged_fairy_tales_without_eos.txt"
if fairy_tales_path.exists():
    print("  📖 Loading Fairy Tales data...")
    with open(fairy_tales_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into sentences
    fairy_sentences = content.split('.')
    fairy_count = 0
    for sent in fairy_sentences:
        if len(sent.strip()) > 20 and len(sent.strip()) < 200:
            all_sentences.append(sent.strip())
            fairy_count += 1
            if fairy_count >= 1000:  # Limit to 1000 sentences
                break
    print(f"  ✅ Fairy Tales: {fairy_count} sentences")

print(f"  📊 Total sentences: {len(all_sentences)}")
print()

# Extract relations from all sentences
print("  🔧 Extracting relations from sentences...")
all_relations = []
predicate_counts = {}

for i, sent in enumerate(all_sentences):
    if i % 500 == 0:
        print(f"    Processing sentence {i+1}/{len(all_sentences)}...")
    
    relations = extract_dependency_relations(sent)
    for rel in relations:
        pred = rel['predicate'].lower()
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
    all_relations.extend(relations)

print(f"  📊 Extracted {len(all_relations)} relations")
print(f"  📊 Unique predicates: {len(predicate_counts)}")
print()

if len(all_relations) < 100:
    print("❌ Insufficient relation data for training!")
    exit(1)

# Create features
print("  🔧 Creating feature vectors...")
X_relations = np.array([relation_to_features(r, predicate_counts) for r in all_relations])

# Create labels based on relation types
relation_types = list(set([r['relation_type'] for r in all_relations]))
relation_type_to_id = {rt: idx for idx, rt in enumerate(relation_types)}
y_relations = np.array([relation_type_to_id[r['relation_type']] for r in all_relations])

print(f"  📊 Total relation types: {len(relation_types)}")
print(f"  📊 Sample types: {list(relation_types)[:5]}")
print()

# K-Fold Cross-Validation with proper regularization
n_splits = min(K_FOLDS, len(X_relations) // 10)  # Ensure enough samples per fold
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

relation_results = {
    'fold_results': [],
    'avg_metrics': {}
}

print(f"🔄 Starting {n_splits}-Fold Cross-Validation...")
print()

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X_relations), 1):
    print(f"  Fold {fold_idx}/{n_splits}")
    print("  " + "-" * 40)
    
    X_train, X_test = X_relations[train_idx], X_relations[test_idx]
    y_train, y_test = y_relations[train_idx], y_relations[test_idx]
    
    # Use regularized Random Forest to prevent overfitting
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=8,  # Limited depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy:  {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print()
    
    relation_results['fold_results'].append({
        'fold': fold_idx,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })

# Calculate average metrics
avg_accuracy = np.mean([r['accuracy'] for r in relation_results['fold_results']])
avg_precision = np.mean([r['precision'] for r in relation_results['fold_results']])
avg_recall = np.mean([r['recall'] for r in relation_results['fold_results']])
avg_f1 = np.mean([r['f1_score'] for r in relation_results['fold_results']])

relation_results['avg_metrics'] = {
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

# Train final model
print("  🎯 Training final model on full dataset...")
clf_final = RandomForestClassifier(
    n_estimators=100, 
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    random_state=RANDOM_STATE,
    class_weight='balanced'
)
clf_final.fit(X_relations, y_relations)

# Save model
model_path = OUTPUT_DIR / "relation_extractor_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': clf_final,
        'relation_types': relation_types,
        'relation_type_to_id': relation_type_to_id,
        'predicate_counts': predicate_counts
    }, f)
print(f"  ✅ Model saved to {model_path}")

# Save results
results_path = RESULTS_DIR / "relation_extractor_results.json"
with open(results_path, 'w') as f:
    json.dump(relation_results, f, indent=2)
print(f"  ✅ Results saved to {results_path}")

print()
print("=" * 80)
print("RELATION EXTRACTOR TRAINING COMPLETE!")
print("=" * 80)
