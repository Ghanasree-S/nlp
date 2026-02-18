"""
Multilingual Training Pipeline for VisualVerse
Trains all NLP models in Hindi, Tamil, and English

Pipeline (same as English):
  1. NLP Preprocessing: SpaCy POS tagging, NER, Dependency Parsing
  2. Text Classification: Narrative (comic) vs Informational (mindmap)
  3. Four NLP Models:
     - Text Classifier (BiLSTM + Attention)
     - Keyphrase Extractor (Gradient Boosting)
     - Topic Modeler (LDA)
     - Relation Extractor (MLP)

Multilingual Support:
  - English: SpaCy en_core_web_sm (full POS/NER/dep)
  - Hindi:   SpaCy xx_ent_wiki_sm + basic fallback
  - Tamil:   SpaCy xx_ent_wiki_sm + basic fallback
"""

import os
import sys
import pickle
import asyncio
from pathlib import Path
from datetime import datetime
from collections import Counter

# Set UTF-8 encoding for output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Import data loader
from data.dataset_loader import DatasetLoader


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def print_metrics(metrics: dict, indent: int = 2):
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_metrics(value, indent + 2)
        elif isinstance(value, float):
            print(" " * indent + f"{key}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 5:
            print(" " * indent + f"{key}: [{value[0]}, {value[1]}, ... ({len(value)} items)]")
        else:
            print(" " * indent + f"{key}: {value}")


def verify_preprocessing(lang_code: str, sample_texts: list):
    """
    Verify NLP preprocessing works for a language.
    Runs SpaCy pipeline and reports POS, NER, dependency parsing stats.
    This confirms the multilingual pipeline uses the same NLP features as English.
    """
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print(f"\n  [NLP] Verifying preprocessing for {lang_names.get(lang_code, lang_code)}...")

    try:
        from nlp.preprocessing.preprocessor import TextPreprocessor
        preprocessor = TextPreprocessor()

        # Process a few sample texts
        samples = sample_texts[:5]
        total_tokens = 0
        total_entities = 0
        total_deps = 0
        total_nouns = 0
        total_chunks = 0
        pos_counts = Counter()
        dep_counts = Counter()
        entity_labels = Counter()

        for text in samples:
            result = preprocessor.process(text, language=lang_code)
            total_tokens += len(result.get("tokens", []))
            total_entities += len(result.get("entities", []))
            total_deps += len(result.get("dependencies", []))
            total_nouns += len(result.get("nouns", []))
            total_chunks += len(result.get("noun_chunks", []))

            for t in result.get("tokens", []):
                pos_counts[t["pos"]] += 1
            for d in result.get("dependencies", []):
                dep_counts[d["dep"]] += 1
            for e in result.get("entities", []):
                entity_labels[e["label"]] += 1

        n = len(samples)
        print(f"    Samples processed: {n}")
        print(f"    Avg tokens/doc:    {total_tokens // n}")
        print(f"    Avg entities/doc:  {total_entities / n:.1f}")
        print(f"    Avg deps/doc:      {total_deps / n:.1f}")
        print(f"    Avg nouns/doc:     {total_nouns // n}")
        print(f"    Avg noun_chunks:   {total_chunks / n:.1f}")
        print(f"    POS tags found:    {dict(pos_counts.most_common(6))}")
        if dep_counts:
            print(f"    Dep labels found:  {dict(dep_counts.most_common(5))}")
        if entity_labels:
            print(f"    Entity types:      {dict(entity_labels.most_common(5))}")
        print(f"    [OK] Preprocessing pipeline verified for {lang_names.get(lang_code, lang_code)}")
        return True

    except Exception as e:
        print(f"    [WARN] Preprocessing check failed: {e}")
        print(f"    Models will still train (preprocessing runs at inference time)")
        return False


def load_multilingual_data(lang_code: str, data_type: str):
    """Load multilingual training data"""
    data_dir = Path("training/data/multilingual")
    filepath = data_dir / f"{data_type}_{lang_code}.pkl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


async def train_text_classifier_multilingual(lang_code: str):
    """Train text classifier for a specific language
    
    Pipeline: Text → BiLSTM + Attention → narrative/informational
    Same architecture as English: Embedding → BiLSTM(2-layer) → Attention → FC
    Classification result determines comic (narrative) or mindmap (informational)
    """
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Text Classifier - {lang_names.get(lang_code, lang_code)}")
    print("Architecture: Embedding -> BiLSTM (2-layer) -> Attention -> FC")
    print("Syllabus: Unit 3 (LSTM, Attention)")
    print("Purpose: Classify text as narrative (comic) or informational (mindmap)")
    print("-" * 40)
    
    try:
        # Load data for the specific language
        if lang_code == "en":
            loader = DatasetLoader()
            loader.load_all()
            texts, labels = loader.prepare_classification_data()
        else:
            data = load_multilingual_data(lang_code, "classifier")
            texts = data["texts"]
            labels = data["labels"]
        
        from nlp.classification.lstm_classifier import AdvancedTextClassifier
        
        classifier = AdvancedTextClassifier()
        classifier._get_training_data = lambda: (texts, labels)
        
        print(f"Training on {len(texts)} examples...")
        print(f"  Narratives (comic):       {sum(labels)}")
        print(f"  Informational (mindmap):  {len(labels) - sum(labels)}")
        
        metrics = await classifier.train()
        
        # Save the trained model with language-specific name
        model_dir = Path("backend/models")
        model_dir.mkdir(exist_ok=True)
        
        if lang_code != "en":
            # Save language-specific model
            import torch
            src_path = model_dir / "lstm_classifier.pt"
            dest_path = model_dir / f"lstm_classifier_{lang_code}.pt"
            if src_path.exists():
                import shutil
                shutil.copy(src_path, dest_path)
                print(f"  Saved model to {dest_path}")
        
        print("\n[OK] Training Complete!")
        print_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"[WARN] BiLSTM failed: {e}")
        
        # Fallback to sklearn classifier (same as English pipeline)
        print(f"\n  Falling back to sklearn RandomForest classifier...")
        try:
            from nlp.classification.classifier import TextClassifier
            
            if lang_code == "en":
                loader = DatasetLoader()
                loader.load_all()
                texts, labels = loader.prepare_classification_data()
            else:
                data = load_multilingual_data(lang_code, "classifier")
                texts = data["texts"]
                labels = data["labels"]
            
            classifier = TextClassifier()
            classifier._get_sample_training_data = lambda: (texts, labels)
            
            metrics = await classifier.train()
            print("\n[OK] Fallback Training Complete!")
            print_metrics(metrics)
            return metrics
        except Exception as e2:
            print(f"[FAIL] Fallback also failed: {e2}")
            import traceback
            traceback.print_exc()
            return {"error": str(e2)}


async def train_keyphrase_extractor_multilingual(lang_code: str):
    """Train keyphrase extractor for a specific language
    
    Pipeline: Text → NLP Preprocessing (POS, NER, noun_chunks) → 
              Candidate Generation → Feature Extraction (11 features) → 
              GradientBoosting Classification
    Uses POS tags, NER entities, and dependency subjects from preprocessing
    """
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Keyphrase Extractor - {lang_names.get(lang_code, lang_code)}")
    print("Architecture: Candidate Generation -> Feature Extraction -> GradientBoosting")
    print("Syllabus: Unit 2 (TF-IDF), Unit 3 (Seq2Seq possible)")
    print("NLP Features: POS tags, NER entities, noun_chunks, dep subjects")
    print("-" * 40)
    
    try:
        # Load data for the specific language
        if lang_code == "en":
            loader = DatasetLoader()
            loader.load_all()
            texts, keyphrases = loader.prepare_keyphrase_data()
        else:
            data = load_multilingual_data(lang_code, "keyphrase")
            texts = data["texts"]
            keyphrases = data["keyphrases"]
        
        from nlp.keyphrase.extractor import KeyphraseExtractor
        
        extractor = KeyphraseExtractor()
        extractor._load_training_data = lambda dataset_name=None: (texts, keyphrases)
        
        print(f"Training on {len(texts)} documents...")
        print(f"  Total keyphrases: {sum(len(kp) for kp in keyphrases)}")
        print(f"  Avg per doc: {sum(len(kp) for kp in keyphrases) / len(texts):.1f}")
        
        metrics = await extractor.train()
        
        # Save language-specific model
        if lang_code != "en":
            model_dir = Path("backend/models")
            model_dir.mkdir(exist_ok=True)
            src_path = model_dir / "keyphrase_model.pkl"
            dest_path = model_dir / f"keyphrase_model_{lang_code}.pkl"
            if src_path.exists():
                import shutil
                shutil.copy(src_path, dest_path)
                print(f"  Saved model to {dest_path}")
        
        print("\n[OK] Training Complete!")
        print_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def train_topic_modeler_multilingual(lang_code: str):
    """Train topic model for a specific language
    
    Pipeline: Text → NLP Preprocessing (lemmatization) → 
              CountVectorizer → LDA → Topic-Word Distributions
    Uses lemmas from preprocessing (stopword-removed base forms)
    """
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Topic Model - {lang_names.get(lang_code, lang_code)}")
    print("Architecture: CountVectorizer -> LDA -> Topic-Word Distributions")
    print("Syllabus: Unit 2 (BoW), Unit 3 (Topic Modeling)")
    print("NLP Features: Lemmas (stopword-removed base forms)")
    print("-" * 40)
    
    try:
        # Load data for the specific language
        if lang_code == "en":
            loader = DatasetLoader()
            loader.load_all()
            texts = loader.prepare_topic_data()
        else:
            data = load_multilingual_data(lang_code, "topic")
            texts = data["texts"]
        
        from nlp.topic_model.topic_modeler import TopicModeler
        
        modeler = TopicModeler(n_topics=10)
        modeler._load_training_data = lambda dataset_name=None: texts
        
        print(f"Training on {len(texts)} documents...")
        
        metrics = await modeler.train()
        
        # Save language-specific model
        if lang_code != "en":
            model_dir = Path("backend/models")
            model_dir.mkdir(exist_ok=True)
            # Save both the model and vectorizer
            for filename in ["topic_model.pkl", "topic_vectorizer.pkl"]:
                src_path = model_dir / filename
                base_name = filename.replace(".pkl", "")
                dest_path = model_dir / f"{base_name}_{lang_code}.pkl"
                if src_path.exists():
                    import shutil
                    shutil.copy(src_path, dest_path)
                    print(f"  Saved {dest_path}")
        
        print("\n[OK] Training Complete!")
        print_metrics(metrics)
        
        # Print discovered topics
        if "topic_words" in metrics:
            print("\n[INFO] Discovered Topics:")
            for topic_id, words in metrics["topic_words"].items():
                print(f"  Topic {topic_id}: {', '.join(words[:5])}")
        
        return metrics
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def train_relation_extractor_multilingual(lang_code: str):
    """Train relation extractor for a specific language
    
    Pipeline: Text → NLP Preprocessing (dependency parsing) →
              Context Encoding (TF-IDF) → Pattern Features → MLP Classification
    Relation Types: IS_A, PART_OF, CAUSES, REQUIRES, RELATES_TO, CONTRASTS, NONE
    Uses dependency parse (nsubj, dobj, pobj, attr) from preprocessing
    """
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Relation Extractor - {lang_names.get(lang_code, lang_code)}")
    print("Architecture: Context Encoding (TF-IDF) -> Pattern Features -> MLP")
    print("Syllabus: Unit 3 (BERT/Transformer or MLP)")
    print("NLP Features: Dependency parse (nsubj, dobj, pobj, attr)")
    print("Relations: IS_A, PART_OF, CAUSES, REQUIRES, RELATES_TO, CONTRASTS, NONE")
    print("-" * 40)
    
    try:
        from nlp.relation.relation_extractor import RelationExtractor
        
        if lang_code != "en":
            # Load multilingual relation data
            data = load_multilingual_data(lang_code, "relation")
            
            # Prepare training data in the format expected by relation extractor
            training_examples = []
            for item in data:
                training_examples.append({
                    "sentence": item["sentence"],
                    "entity1": item["entity1"],
                    "entity2": item["entity2"],
                    "relation": item["relation"]
                })
            
            extractor = RelationExtractor()
            # Inject the multilingual training data
            extractor._get_sample_training_data = lambda: training_examples
        else:
            extractor = RelationExtractor()
        
        print(f"Training on relation extraction data...")
        
        metrics = await extractor.train()
        
        # Save language-specific model
        if lang_code != "en":
            model_dir = Path("backend/models")
            model_dir.mkdir(exist_ok=True)
            src_path = model_dir / "relation_model.pkl"
            dest_path = model_dir / f"relation_model_{lang_code}.pkl"
            if src_path.exists():
                import shutil
                shutil.copy(src_path, dest_path)
                print(f"  Saved model to {dest_path}")
        
        print("\n[OK] Training Complete!")
        print_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def train_language(lang_code: str):
    """Train all models for a specific language
    
    Same pipeline as English:
      Step 0: Verify NLP preprocessing (POS, NER, dependency parsing)
      Step 1: Train Text Classifier (comic vs mindmap decision)
      Step 2: Train Keyphrase Extractor (uses POS, NER, noun_chunks)
      Step 3: Train Topic Modeler (uses lemmas from preprocessing)
      Step 4: Train Relation Extractor (uses dependency parse)
    """
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    lang_name = lang_names.get(lang_code, lang_code)
    
    print("\n" + "=" * 70)
    print(f" Training All Models for {lang_name} ({lang_code})")
    print("=" * 70)
    print(f"\n Pipeline: NLP Preprocessing -> Classification -> 4 Models")
    print(f" Preprocessing: SpaCy POS tagging, NER, Dependency Parsing")
    print(f" Classification: narrative (comic) vs informational (mindmap)")
    print(f" Models: Classifier, Keyphrase, Topic Model, Relation Extractor")
    
    all_metrics = {}
    
    # Step 0: Verify NLP preprocessing works for this language
    try:
        if lang_code == "en":
            sample_texts = ["Machine learning is a subset of artificial intelligence.",
                           "Once upon a time, there was a brave knight."]
        else:
            data = load_multilingual_data(lang_code, "classifier")
            sample_texts = data["texts"][:5]
        verify_preprocessing(lang_code, sample_texts)
    except Exception as e:
        print(f"  [WARN] Could not verify preprocessing: {e}")
    
    # Step 1: Train Text Classifier
    try:
        metrics = await train_text_classifier_multilingual(lang_code)
        all_metrics["classifier"] = metrics
    except Exception as e:
        print(f"[FAIL] Classifier failed: {e}")
        all_metrics["classifier"] = {"error": str(e)}
    
    # Step 2: Train Keyphrase Extractor (uses POS, NER, noun_chunks)
    try:
        metrics = await train_keyphrase_extractor_multilingual(lang_code)
        all_metrics["keyphrase"] = metrics
    except Exception as e:
        print(f"[FAIL] Keyphrase extractor failed: {e}")
        all_metrics["keyphrase"] = {"error": str(e)}
    
    # Step 3: Train Topic Modeler (uses lemmas from preprocessing)
    try:
        metrics = await train_topic_modeler_multilingual(lang_code)
        all_metrics["topic_model"] = metrics
    except Exception as e:
        print(f"[FAIL] Topic modeler failed: {e}")
        all_metrics["topic_model"] = {"error": str(e)}
    
    # Step 4: Train Relation Extractor (uses dependency parse)
    try:
        metrics = await train_relation_extractor_multilingual(lang_code)
        all_metrics["relation"] = metrics
    except Exception as e:
        print(f"[FAIL] Relation extractor failed: {e}")
        all_metrics["relation"] = {"error": str(e)}
    
    # Per-language summary
    print(f"\n  --- {lang_name} Summary ---")
    for model_name, metrics in all_metrics.items():
        if isinstance(metrics, dict) and "error" in metrics:
            print(f"  [FAIL] {model_name}")
        else:
            acc = metrics.get('accuracy', metrics.get('f1_score', metrics.get('coherence', 'N/A'))) if isinstance(metrics, dict) else 'N/A'
            print(f"  [OK]   {model_name}: {acc}")
    
    return all_metrics


async def main():
    """Main multilingual training pipeline
    
    Same pipeline as English (train_all.py) applied to all languages:
      1. NLP Preprocessing: SpaCy POS, NER, Dependency Parsing
      2. Classification: narrative (comic) vs informational (mindmap)  
      3. Four Models: Classifier, Keyphrase, Topic Model, Relation Extractor
    """
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print(" VisualVerse - Multilingual NLP Training Pipeline")
    print(" Languages: English | Hindi | Tamil")
    print(f" Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print("\n Pipeline (same as English):")
    print("   Step 0: NLP Preprocessing (SpaCy POS, NER, Dep Parsing)")
    print("   Step 1: Text Classifier  (BiLSTM+Attention -> comic/mindmap)")
    print("   Step 2: Keyphrase Extractor (GradientBoosting, uses POS/NER)")
    print("   Step 3: Topic Modeler (LDA, uses lemmas)")
    print("   Step 4: Relation Extractor (MLP, uses dep parse)")
    print("=" * 70)
    
    languages = ["en", "hi", "ta"]
    all_results = {}
    
    # Train models for each language
    for lang in languages:
        try:
            metrics = await train_language(lang)
            all_results[lang] = metrics
        except Exception as e:
            print(f"\n[FAIL] Failed to train {lang}: {e}")
            all_results[lang] = {"error": str(e)}
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final Summary
    print("\n" + "=" * 70)
    print(" MULTILINGUAL TRAINING SUMMARY")
    print("=" * 70)
    
    model_names_display = {
        "classifier": "Text Classifier (BiLSTM+Attention)",
        "keyphrase": "Keyphrase Extractor (GradientBoosting)",
        "topic_model": "Topic Modeler (LDA)",
        "relation": "Relation Extractor (MLP)"
    }
    
    total_success = 0
    total_fail = 0
    
    for lang_code, lang_metrics in all_results.items():
        lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
        lang_name = lang_names.get(lang_code, lang_code)
        
        print(f"\n  {lang_name} ({lang_code}):")
        if isinstance(lang_metrics, dict) and "error" in lang_metrics:
            print(f"    [FAIL] All models failed: {lang_metrics['error']}")
            total_fail += 4
            continue
            
        for model_name, metrics in lang_metrics.items():
            display_name = model_names_display.get(model_name, model_name)
            if isinstance(metrics, dict) and "error" in metrics:
                print(f"    [FAIL] {display_name}: {metrics['error'][:50]}")
                total_fail += 1
            else:
                acc = metrics.get('accuracy', metrics.get('f1_score', metrics.get('coherence', 'N/A'))) if isinstance(metrics, dict) else 'N/A'
                print(f"    [OK]   {display_name}: {acc}")
                total_success += 1
    
    print(f"\n  Results: {total_success} succeeded, {total_fail} failed")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
    
    # Save results
    results_path = Path("training/multilingual_training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({
            "results": all_results,
            "timestamp": start_time.isoformat(),
            "duration_seconds": duration,
            "languages": languages,
            "pipeline": "NLP Preprocessing -> Classification -> 4 Models"
        }, f)
    print(f"  Results saved to {results_path}")
    
    print(f"\n[OK] Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    asyncio.run(main())

