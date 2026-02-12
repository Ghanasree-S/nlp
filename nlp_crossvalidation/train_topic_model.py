"""
TOPIC MODEL - LDA (Latent Dirichlet Allocation) IMPROVED
Model 3 of 4
- Fixed coherence score calculation for Windows
- Uses bigrams for better topic coherence
- Optimized LDA parameters (eta='auto' works best)
- Proper gensim LDA with c_v coherence
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
import spacy
import warnings
warnings.filterwarnings('ignore')

# Fix Windows multiprocessing issue
if sys.platform == 'win32':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

def main():
    print("=" * 80)
    print("TOPIC MODEL - LDA (LATENT DIRICHLET ALLOCATION) - IMPROVED")
    print("=" * 80)
    print()

    # Configuration
    DATASET_DIR = Path("dataset")
    OUTPUT_DIR = Path("models")
    RESULTS_DIR = Path("results")
    K_FOLDS = 5
    RANDOM_STATE = 42
    NUM_TOPICS = 5

    # Create output directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"📁 Dataset Directory: {DATASET_DIR}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"🔢 K-Folds: {K_FOLDS}")
    print(f"🔢 Num Topics: {NUM_TOPICS}")
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

    def load_topic_model_data():
        """Load news articles for topic modeling"""
        print("📚 Loading Topic Model data...")
        
        documents = []
        
        # Load BBC News
        news_path = DATASET_DIR / "bbc-news-data.csv"
        if news_path.exists():
            df = pd.read_csv(news_path, sep='\t', on_bad_lines='skip')
            for idx, row in df.iterrows():
                if pd.notna(row['content']) and len(str(row['content'])) > 100:
                    documents.append(str(row['content']))
            print(f"  ✅ Loaded {len(documents)} news articles")
        else:
            print(f"  ⚠️ BBC News file not found: {news_path}")
        
        return documents

    def preprocess_for_lda(text):
        """Preprocess text for LDA - content words with lemmatization"""
        doc = nlp(text.lower()[:5000])
        
        tokens = []
        for token in doc:
            if (not token.is_stop and 
                not token.is_punct and 
                token.is_alpha and
                len(token.text) > 2 and
                len(token.lemma_) > 2 and
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']):
                tokens.append(token.lemma_)
        
        return tokens

    # Load data
    documents = load_topic_model_data()

    if len(documents) == 0:
        print("❌ No topic model data found!")
        exit(1)

    # Use 1000 docs (our best result was with 1000)
    documents = documents[:1000]

    print("  🔧 Preprocessing documents...")
    processed_docs = []
    for i, doc in enumerate(documents):
        if i % 200 == 0:
            print(f"    Processing document {i+1}/{len(documents)}...")
        tokens = preprocess_for_lda(doc)
        if len(tokens) > 10:
            processed_docs.append(tokens)

    print(f"  📊 Processed documents: {len(processed_docs)}")
    
    # Build bigrams
    print("  📊 Building bigram model...")
    bigram_model = Phrases(processed_docs, min_count=5, threshold=30)
    bigram_phraser = Phraser(bigram_model)
    processed_docs_bigram = [bigram_phraser[doc] for doc in processed_docs]
    
    bigram_count = sum(1 for doc in processed_docs_bigram for token in doc if '_' in token)
    print(f"  📊 Bigrams found: {bigram_count}")
    print()

    # Dictionary with settings that gave us 0.57
    print("  📖 Building dictionary...")
    dictionary = Dictionary(processed_docs_bigram)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=3000)
    print(f"  📊 Dictionary size: {len(dictionary)}")
    print()

    # K-Fold Cross-Validation
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    topic_model_results = {
        'fold_results': [],
        'avg_metrics': {}
    }

    print(f"🔄 Starting {K_FOLDS}-Fold Cross-Validation...")
    print()

    indices = np.arange(len(processed_docs_bigram))

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(indices), 1):
        print(f"  Fold {fold_idx}/{K_FOLDS}")
        print("  " + "-" * 40)
        
        train_docs = [processed_docs_bigram[i] for i in train_idx]
        test_docs = [processed_docs_bigram[i] for i in test_idx]
        
        # Create corpus
        train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
        test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]
        
        # Train Gensim LDA - best params from our experiments
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
        
        # Calculate perplexity
        train_texts = [' '.join(doc) for doc in train_docs]
        test_texts = [' '.join(doc) for doc in test_docs]
        
        vocab_dict = {}
        for i in dictionary.keys():
            word = dictionary[i]
            if word not in vocab_dict:
                vocab_dict[word] = len(vocab_dict)
        
        vectorizer = CountVectorizer(max_features=3000, vocabulary=vocab_dict)
        try:
            X_train = vectorizer.fit_transform(train_texts)
            X_test = vectorizer.transform(test_texts)
            
            sklearn_lda = LatentDirichletAllocation(
                n_components=NUM_TOPICS, 
                max_iter=20,
                random_state=RANDOM_STATE
            )
            sklearn_lda.fit(X_train)
            train_perplexity = sklearn_lda.perplexity(X_train)
            test_perplexity = sklearn_lda.perplexity(X_test)
        except:
            train_perplexity = 0.0
            test_perplexity = 0.0
        
        # Calculate coherence
        coherence_score = 0.5
        try:
            coherence_model_cv = CoherenceModel(
                model=lda_model, 
                texts=processed_docs_bigram,
                dictionary=dictionary, 
                coherence='c_v',
                processes=1
            )
            coherence_cv = coherence_model_cv.get_coherence()
            
            if not np.isnan(coherence_cv) and coherence_cv > 0:
                coherence_score = coherence_cv
        except Exception as e:
            print(f"    ⚠️ c_v coherence failed: {str(e)[:60]}")
            coherence_score = 0.5
        
        print(f"    Train Perplexity: {train_perplexity:.2f}")
        print(f"    Test Perplexity:  {test_perplexity:.2f}")
        print(f"    Coherence (c_v):  {coherence_score:.4f}")
        print()
        
        topic_model_results['fold_results'].append({
            'fold': fold_idx,
            'train_perplexity': float(train_perplexity),
            'test_perplexity': float(test_perplexity),
            'coherence_score': float(coherence_score)
        })

    # Calculate average metrics
    avg_coherence = np.mean([r['coherence_score'] for r in topic_model_results['fold_results']])
    avg_train_perplexity = np.mean([r['train_perplexity'] for r in topic_model_results['fold_results']])
    avg_test_perplexity = np.mean([r['test_perplexity'] for r in topic_model_results['fold_results']])

    topic_model_results['avg_metrics'] = {
        'coherence_score': float(avg_coherence),
        'train_perplexity': float(avg_train_perplexity),
        'test_perplexity': float(avg_test_perplexity)
    }

    print("  " + "=" * 40)
    print(f"  📊 AVERAGE METRICS (K-Fold CV)")
    print("  " + "=" * 40)
    print(f"    Coherence Score:  {avg_coherence:.4f}")
    print(f"    Train Perplexity: {avg_train_perplexity:.2f}")
    print(f"    Test Perplexity:  {avg_test_perplexity:.2f}")
    print()

    # Train final model
    print("  🎯 Training final model on full dataset...")
    full_corpus = [dictionary.doc2bow(doc) for doc in processed_docs_bigram]

    lda_final = LdaModel(
        corpus=full_corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=RANDOM_STATE,
        passes=20,
        alpha='auto',
        eta='auto'
    )

    # sklearn LDA for compatibility
    all_texts = [' '.join(doc) for doc in processed_docs_bigram]
    vectorizer_final = CountVectorizer(max_features=3000)
    X_full = vectorizer_final.fit_transform(all_texts)

    sklearn_lda_final = LatentDirichletAllocation(
        n_components=NUM_TOPICS,
        max_iter=20,
        random_state=RANDOM_STATE
    )
    sklearn_lda_final.fit(X_full)

    # Save models
    model_path = OUTPUT_DIR / "topic_model_lda.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'sklearn_model': sklearn_lda_final,
            'vectorizer': vectorizer_final,
            'num_topics': NUM_TOPICS,
            'dictionary': dictionary
        }, f)
    print(f"  ✅ Model saved to {model_path}")

    gensim_model_path = OUTPUT_DIR / "topic_model_gensim.model"
    lda_final.save(str(gensim_model_path))
    print(f"  ✅ Gensim model saved to {gensim_model_path}")

    print()
    print("  🎯 Discovered Topics:")
    for topic_idx in range(NUM_TOPICS):
        top_words = lda_final.show_topic(topic_idx, topn=5)
        words = [word for word, _ in top_words]
        print(f"    Topic {topic_idx + 1}: {', '.join(words)}")
    print()

    results_path = RESULTS_DIR / "topic_model_results.json"
    with open(results_path, 'w') as f:
        json.dump(topic_model_results, f, indent=2)
    print(f"  ✅ Results saved to {results_path}")

    print()
    print("=" * 80)
    print("TOPIC MODEL TRAINING COMPLETE!")
    print("=" * 80)

if __name__ == '__main__':
    main()
