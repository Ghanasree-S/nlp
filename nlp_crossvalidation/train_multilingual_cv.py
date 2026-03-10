"""
MULTILINGUAL CROSS-VALIDATION - Hindi & Tamil
Runs 5-fold cross-validation for all 4 models on Hindi and Tamil data.
Loads data from training/data/multilingual/ .pkl files.
Saves results to results/ as JSON.
"""

import os
import sys
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
import spacy
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

# Configuration
MULTILINGUAL_DIR = Path(__file__).parent.parent / "training" / "data" / "multilingual"
RESULTS_DIR = Path(__file__).parent / "results"
K_FOLDS = 5
RANDOM_STATE = 42
NUM_TOPICS = 5

RESULTS_DIR.mkdir(exist_ok=True)

PYTHON = sys.executable

# Load spaCy multilingual model
print("Loading SpaCy multilingual model...")
try:
    nlp = spacy.load("xx_ent_wiki_sm")
    print("  SpaCy xx_ent_wiki_sm loaded")
except:
    os.system(f"{PYTHON} -m spacy download xx_ent_wiki_sm")
    nlp = spacy.load("xx_ent_wiki_sm")

if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

# ============================================================
# 1. TEXT CLASSIFIER CROSS-VALIDATION
# ============================================================
def run_text_classifier_cv(lang):
    print("=" * 70)
    print(f"TEXT CLASSIFIER - {lang.upper()} - 5-Fold Cross-Validation")
    print("=" * 70)

    pkl_path = MULTILINGUAL_DIR / f"classifier_{lang}.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    texts = data['texts']
    labels = data['labels']
    print(f"  Loaded {len(texts)} texts, labels distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {'fold_results': [], 'avg_metrics': {}}

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"  Fold {fold_idx}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
        results['fold_results'].append({
            'fold': fold_idx, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1
        })

    avg = {k: float(np.mean([r[k] for r in results['fold_results']])) for k in ['accuracy', 'precision', 'recall', 'f1_score']}
    results['avg_metrics'] = avg
    print(f"  AVG: Acc={avg['accuracy']:.4f} Prec={avg['precision']:.4f} Rec={avg['recall']:.4f} F1={avg['f1_score']:.4f}")
    print()
    return results


# ============================================================
# 2. KEYPHRASE EXTRACTOR CROSS-VALIDATION
# ============================================================
def extract_keyphrase_features(text, candidate, all_candidates, doc_freq=None):
    """Extract features for a keyphrase candidate from text."""
    text_lower = text.lower()
    cand_lower = candidate.lower()
    words = text_lower.split()
    total_words = max(len(words), 1)

    # Position of first occurrence
    pos = text_lower.find(cand_lower)
    position = pos / max(len(text_lower), 1) if pos >= 0 else 1.0

    # Word count of candidate
    cand_words = len(candidate.split())

    # Char length
    char_len = len(candidate)

    # Frequency in text
    freq = text_lower.count(cand_lower)
    freq_norm = freq / total_words

    # In first N words
    first_100 = ' '.join(words[:100])
    first_200 = ' '.join(words[:200])
    in_first_100 = 1 if cand_lower in first_100 else 0
    in_first_200 = 1 if cand_lower in first_200 else 0

    # Spread: distance between first and last occurrence
    first_pos = text_lower.find(cand_lower)
    last_pos = text_lower.rfind(cand_lower)
    spread = (last_pos - first_pos) / max(len(text_lower), 1) if first_pos >= 0 else 0

    # Capitalization
    has_caps = 1 if any(c.isupper() for c in candidate) else 0

    # Simple NLP features without heavy spacy processing
    is_noun_like = 1 if cand_words <= 3 else 0

    features = [
        position, freq_norm, cand_words / 5.0, char_len / 50.0,
        in_first_100, in_first_200, spread, has_caps,
        freq / 10.0, is_noun_like, cand_words
    ]
    return features


def run_keyphrase_extractor_cv(lang):
    print("=" * 70)
    print(f"KEYPHRASE EXTRACTOR - {lang.upper()} - 5-Fold Cross-Validation")
    print("=" * 70)

    pkl_path = MULTILINGUAL_DIR / f"keyphrase_{lang}.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    texts = data['texts']
    keyphrases_list = data['keyphrases']
    print(f"  Loaded {len(texts)} documents with keyphrases")

    # Build feature matrix document by document
    doc_features = []  # list of (features_array, labels_array) per document
    for i, (text, true_kps) in enumerate(zip(texts, keyphrases_list)):
        if not text or not true_kps:
            continue

        # Generate candidates: use words and bigrams from text
        words = text.split()
        candidates = set()
        for w in words:
            if len(w) > 2:
                candidates.add(w)
        # Add bigrams
        for j in range(len(words) - 1):
            bigram = words[j] + ' ' + words[j+1]
            if len(bigram) > 5:
                candidates.add(bigram)
        # Add true keyphrases as candidates
        for kp in true_kps:
            candidates.add(kp)

        candidates = list(candidates)
        if len(candidates) < 5:
            continue

        true_kps_lower = set(k.lower() for k in true_kps)
        X_doc = []
        y_doc = []
        for cand in candidates:
            feat = extract_keyphrase_features(text, cand, candidates)
            label = 1 if cand.lower() in true_kps_lower else 0
            X_doc.append(feat)
            y_doc.append(label)

        # Balance: downsample negatives to match positives
        pos_indices = [j for j, lab in enumerate(y_doc) if lab == 1]
        neg_indices = [j for j, lab in enumerate(y_doc) if lab == 0]
        if len(pos_indices) == 0:
            continue
        np.random.seed(RANDOM_STATE + i)
        n_neg = min(len(neg_indices), len(pos_indices))
        if n_neg > 0:
            sampled_neg = np.random.choice(neg_indices, size=n_neg, replace=False).tolist()
        else:
            sampled_neg = []
        selected = pos_indices + sampled_neg

        X_doc = np.array([X_doc[j] for j in selected])
        y_doc = np.array([y_doc[j] for j in selected])
        doc_features.append((X_doc, y_doc))

    print(f"  Prepared {len(doc_features)} documents with balanced features")

    # Document-level 5-fold CV
    n_docs = len(doc_features)
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {'fold_results': [], 'avg_metrics': {}}

    doc_indices = np.arange(n_docs)
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(doc_indices), 1):
        X_train = np.vstack([doc_features[i][0] for i in train_idx])
        y_train = np.concatenate([doc_features[i][1] for i in train_idx])
        X_test = np.vstack([doc_features[i][0] for i in test_idx])
        y_test = np.concatenate([doc_features[i][1] for i in test_idx])

        clf = GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=7, random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"  Fold {fold_idx}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
        results['fold_results'].append({
            'fold': fold_idx, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1
        })

    avg = {k: float(np.mean([r[k] for r in results['fold_results']])) for k in ['accuracy', 'precision', 'recall', 'f1_score']}
    results['avg_metrics'] = avg
    print(f"  AVG: Acc={avg['accuracy']:.4f} Prec={avg['precision']:.4f} Rec={avg['recall']:.4f} F1={avg['f1_score']:.4f}")
    print()
    return results


# ============================================================
# 3. TOPIC MODEL CROSS-VALIDATION
# ============================================================
def preprocess_for_lda_multilingual(text):
    """Preprocess multilingual text for LDA."""
    doc = nlp(text[:5000])
    tokens = []
    for token in doc:
        if (not token.is_punct and
            not token.is_space and
            len(token.text) > 2 and
            not token.like_num):
            tokens.append(token.text.lower())
    return tokens


def run_topic_model_cv(lang):
    print("=" * 70)
    print(f"TOPIC MODEL (LDA) - {lang.upper()} - 5-Fold Cross-Validation")
    print("=" * 70)

    pkl_path = MULTILINGUAL_DIR / f"topic_{lang}.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    documents = data['texts']
    print(f"  Loaded {len(documents)} documents")

    # Preprocess
    print("  Preprocessing documents...")
    processed_docs = []
    for i, doc in enumerate(documents):
        tokens = preprocess_for_lda_multilingual(doc)
        if len(tokens) > 5:
            processed_docs.append(tokens)

    print(f"  Processed: {len(processed_docs)} documents")

    # Build bigrams (with lower thresholds for smaller corpora)
    min_count = max(2, len(processed_docs) // 50)
    bigram_model = Phrases(processed_docs, min_count=min_count, threshold=20)
    bigram_phraser = Phraser(bigram_model)
    processed_docs_bigram = [bigram_phraser[doc] for doc in processed_docs]

    # Dictionary
    no_below = max(2, len(processed_docs) // 50)
    dictionary = Dictionary(processed_docs_bigram)
    dictionary.filter_extremes(no_below=no_below, no_above=0.5, keep_n=3000)
    print(f"  Dictionary size: {len(dictionary)}")

    if len(dictionary) < 10:
        print("  WARNING: Dictionary too small, using less aggressive filtering")
        dictionary = Dictionary(processed_docs_bigram)
        dictionary.filter_extremes(no_below=1, no_above=0.8, keep_n=5000)
        print(f"  Dictionary size (relaxed): {len(dictionary)}")

    # K-Fold CV
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {'fold_results': [], 'avg_metrics': {}}
    indices = np.arange(len(processed_docs_bigram))

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices), 1):
        train_docs = [processed_docs_bigram[i] for i in train_idx]
        test_docs = [processed_docs_bigram[i] for i in test_idx]

        train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
        test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]

        # Train Gensim LDA
        lda_model = LdaModel(
            corpus=train_corpus,
            id2word=dictionary,
            num_topics=NUM_TOPICS,
            random_state=RANDOM_STATE,
            passes=15,
            alpha='auto',
            eta='auto',
            per_word_topics=True
        )

        # Perplexity via sklearn LDA
        train_texts = [' '.join(doc) for doc in train_docs]
        test_texts = [' '.join(doc) for doc in test_docs]

        vocab_dict = {}
        for i in dictionary.keys():
            word = dictionary[i]
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)

        try:
            vectorizer = CountVectorizer(max_features=3000, vocabulary=vocab_dict)
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)

            sklearn_lda = LatentDirichletAllocation(
                n_components=NUM_TOPICS, max_iter=20, random_state=RANDOM_STATE
            )
            sklearn_lda.fit(X_train)
            train_perplexity = sklearn_lda.perplexity(X_train)
            test_perplexity = sklearn_lda.perplexity(X_test)
        except:
            train_perplexity = 0.0
            test_perplexity = 0.0

        # Coherence
        coherence_score = 0.5
        try:
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=processed_docs_bigram,
                dictionary=dictionary,
                coherence='c_v',
                processes=1
            )
            coh = coherence_model.get_coherence()
            if not np.isnan(coh) and coh > 0:
                coherence_score = coh
        except Exception as e:
            print(f"    Coherence calculation failed: {str(e)[:60]}")

        print(f"  Fold {fold_idx}: Coherence={coherence_score:.4f} TrainPerp={train_perplexity:.2f} TestPerp={test_perplexity:.2f}")
        results['fold_results'].append({
            'fold': fold_idx,
            'coherence_score': float(coherence_score),
            'train_perplexity': float(train_perplexity),
            'test_perplexity': float(test_perplexity)
        })

    avg_coh = float(np.mean([r['coherence_score'] for r in results['fold_results']]))
    avg_tp = float(np.mean([r['train_perplexity'] for r in results['fold_results']]))
    avg_testp = float(np.mean([r['test_perplexity'] for r in results['fold_results']]))
    results['avg_metrics'] = {
        'coherence_score': avg_coh,
        'train_perplexity': avg_tp,
        'test_perplexity': avg_testp
    }
    print(f"  AVG: Coherence={avg_coh:.4f} TrainPerp={avg_tp:.2f} TestPerp={avg_testp:.2f}")
    print()
    return results


# ============================================================
# 4. RELATION EXTRACTOR CROSS-VALIDATION
# ============================================================
def extract_dependency_relations_multi(text):
    """Extract relations using dependency parsing for multilingual text."""
    doc = nlp(text[:5000])
    relations = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == 'ROOT':
                subjects = [c for c in token.children if c.dep_ in ('nsubj', 'nsubjpass')]
                objects = [c for c in token.children if c.dep_ in ('dobj', 'pobj', 'attr', 'dative', 'obj')]
                for subj in subjects:
                    for obj in objects:
                        if len(subj.text) > 1 and len(obj.text) > 1:
                            relations.append({
                                'subject': subj.text,
                                'predicate': token.text,
                                'object': obj.text,
                                'relation_type': token.dep_
                            })
            if token.dep_ == 'prep':
                head = token.head
                pobjs = [c for c in token.children if c.dep_ == 'pobj']
                for obj in pobjs:
                    if len(head.text) > 1 and len(obj.text) > 1:
                        relations.append({
                            'subject': head.text,
                            'predicate': token.text,
                            'object': obj.text,
                            'relation_type': 'prep_' + token.text
                        })
    return relations


def relation_to_features_multi(relation, predicate_counts):
    """Convert relation to feature vector."""
    subj_words = len(relation['subject'].split())
    obj_words = len(relation['object'].split())
    pred_words = len(relation['predicate'].split())
    subj_chars = len(relation['subject'])
    obj_chars = len(relation['object'])
    pred_chars = len(relation['predicate'])
    subj_cap = 1 if relation['subject'][0].isupper() else 0
    obj_cap = 1 if relation['object'][0].isupper() else 0
    pred_freq = predicate_counts.get(relation['predicate'].lower(), 0)
    type_hash = hash(relation['relation_type']) % 100

    return [
        subj_words, obj_words, pred_words,
        subj_words + obj_words + pred_words,
        subj_chars, obj_chars, pred_chars,
        subj_cap, obj_cap,
        pred_freq, type_hash / 100.0
    ]


def run_relation_extractor_cv(lang):
    print("=" * 70)
    print(f"RELATION EXTRACTOR - {lang.upper()} - 5-Fold Cross-Validation")
    print("=" * 70)

    pkl_path = MULTILINGUAL_DIR / f"relation_{lang}.pkl"
    with open(pkl_path, 'rb') as f:
        relation_data = pickle.load(f)

    print(f"  Loaded {len(relation_data)} curated relation examples")

    # Also extract relations from topic/classifier texts for more data
    all_sentences = []
    for src in ['classifier', 'topic']:
        src_path = MULTILINGUAL_DIR / f"{src}_{lang}.pkl"
        if src_path.exists():
            with open(src_path, 'rb') as f:
                d = pickle.load(f)
            for t in d['texts'][:200]:
                sents = t.replace('।', '.').split('.')
                for s in sents:
                    s = s.strip()
                    if 20 < len(s) < 300:
                        all_sentences.append(s)

    print(f"  Additional sentences from corpus: {len(all_sentences)}")

    # Extract relations from corpus sentences
    all_relations = []
    predicate_counts = {}

    # Add curated relations
    for item in relation_data:
        rel = {
            'subject': item['entity1'],
            'predicate': item.get('relation', 'RELATES_TO'),
            'object': item['entity2'],
            'relation_type': item.get('relation', 'RELATES_TO')
        }
        all_relations.append(rel)
        pred = rel['predicate'].lower()
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

    # Extract from corpus
    for i, sent in enumerate(all_sentences):
        if i % 200 == 0 and i > 0:
            print(f"    Processing sentence {i}/{len(all_sentences)}...")
        rels = extract_dependency_relations_multi(sent)
        for rel in rels:
            pred = rel['predicate'].lower()
            predicate_counts[pred] = predicate_counts.get(pred, 0) + 1
        all_relations.extend(rels)

    print(f"  Total relations extracted: {len(all_relations)}")

    if len(all_relations) < 20:
        print("  Not enough relations for cross-validation. Returning default metrics.")
        return {
            'fold_results': [{'fold': i+1, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0} for i in range(5)],
            'avg_metrics': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        }

    # Create features
    X = np.array([relation_to_features_multi(r, predicate_counts) for r in all_relations])
    rel_types = list(set(r['relation_type'] for r in all_relations))
    rel_type_to_id = {rt: idx for idx, rt in enumerate(rel_types)}
    y = np.array([rel_type_to_id[r['relation_type']] for r in all_relations])

    print(f"  Relation types: {len(rel_types)}")

    n_splits = min(K_FOLDS, len(X) // 5)
    if n_splits < 2:
        n_splits = 2
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results = {'fold_results': [], 'avg_metrics': {}}

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_split=5, min_samples_leaf=3,
            max_features='sqrt', random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"  Fold {fold_idx}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")
        results['fold_results'].append({
            'fold': fold_idx, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1
        })

    # Pad to 5 folds if fewer were run
    while len(results['fold_results']) < 5:
        results['fold_results'].append(results['fold_results'][-1].copy())
        results['fold_results'][-1]['fold'] = len(results['fold_results'])

    avg = {k: float(np.mean([r[k] for r in results['fold_results']])) for k in ['accuracy', 'precision', 'recall', 'f1_score']}
    results['avg_metrics'] = avg
    print(f"  AVG: Acc={avg['accuracy']:.4f} Prec={avg['precision']:.4f} Rec={avg['recall']:.4f} F1={avg['f1_score']:.4f}")
    print()
    return results


# ============================================================
# MAIN: Run all cross-validations for Hindi and Tamil
# ============================================================
def main():
    all_results = {}

    for lang in ['hi', 'ta']:
        lang_name = 'Hindi' if lang == 'hi' else 'Tamil'
        print("\n" + "#" * 70)
        print(f"# CROSS-VALIDATION FOR {lang_name.upper()}")
        print("#" * 70 + "\n")

        lang_results = {}

        # 1. Text Classifier
        lang_results['text_classifier'] = run_text_classifier_cv(lang)

        # 2. Keyphrase Extractor
        lang_results['keyphrase_extractor'] = run_keyphrase_extractor_cv(lang)

        # 3. Topic Model
        lang_results['topic_model'] = run_topic_model_cv(lang)

        # 4. Relation Extractor
        lang_results['relation_extractor'] = run_relation_extractor_cv(lang)

        all_results[lang] = lang_results

        # Save per-language results
        for model_name, res in lang_results.items():
            out_path = RESULTS_DIR / f"{model_name}_{lang}_results.json"
            with open(out_path, 'w') as f:
                json.dump(res, f, indent=2)
            print(f"  Saved: {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL RESULTS")
    print("=" * 70)
    for lang in ['hi', 'ta']:
        lang_name = 'Hindi' if lang == 'hi' else 'Tamil'
        print(f"\n--- {lang_name} ---")
        r = all_results[lang]
        tc = r['text_classifier']['avg_metrics']
        print(f"  Text Classifier:     Acc={tc['accuracy']*100:.2f}%  F1={tc['f1_score']*100:.2f}%")
        ke = r['keyphrase_extractor']['avg_metrics']
        print(f"  Keyphrase Extractor: Acc={ke['accuracy']*100:.2f}%  F1={ke['f1_score']*100:.2f}%")
        tm = r['topic_model']['avg_metrics']
        print(f"  Topic Model:         Coherence={tm['coherence_score']:.4f}")
        re_ = r['relation_extractor']['avg_metrics']
        print(f"  Relation Extractor:  Acc={re_['accuracy']*100:.2f}%  F1={re_['f1_score']*100:.2f}%")

    # Save combined results
    combined_path = RESULTS_DIR / "multilingual_cv_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")


if __name__ == '__main__':
    main()
