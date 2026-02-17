"""
Multilingual Training Pipeline for VisualVerse
Trains all NLP models in Hindi, Tamil, and English
"""

import os
import sys
import pickle
import asyncio
from pathlib import Path
from datetime import datetime

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


def load_multilingual_data(lang_code: str, data_type: str):
    """Load multilingual training data"""
    data_dir = Path("training/data/multilingual")
    filepath = data_dir / f"{data_type}_{lang_code}.pkl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


async def train_text_classifier_multilingual(lang_code: str):
    """Train text classifier for a specific language"""
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Text Classifier - {lang_names.get(lang_code, lang_code)}")
    print("Architecture: BiLSTM + Attention")
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
        print(f"  Narratives: {sum(labels)}")
        print(f"  Informational: {len(labels) - sum(labels)}")
        
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
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


async def train_keyphrase_extractor_multilingual(lang_code: str):
    """Train keyphrase extractor for a specific language"""
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Keyphrase Extractor - {lang_names.get(lang_code, lang_code)}")
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
    """Train topic model for a specific language"""
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Topic Model - {lang_names.get(lang_code, lang_code)}")
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
    """Train relation extractor for a specific language"""
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    print_header(f"Training Relation Extractor - {lang_names.get(lang_code, lang_code)}")
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
    """Train all models for a specific language"""
    lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
    lang_name = lang_names.get(lang_code, lang_code)
    
    print("\n" + "=" * 70)
    print(f" Training All Models for {lang_name} ({lang_code})")
    print("=" * 70)
    
    all_metrics = {}
    
    # 1. Train Text Classifier
    try:
        metrics = await train_text_classifier_multilingual(lang_code)
        all_metrics["classifier"] = metrics
    except Exception as e:
        print(f"[FAIL] Classifier failed: {e}")
        all_metrics["classifier"] = {"error": str(e)}
    
    # 2. Train Keyphrase Extractor
    try:
        metrics = await train_keyphrase_extractor_multilingual(lang_code)
        all_metrics["keyphrase"] = metrics
    except Exception as e:
        print(f"[FAIL] Keyphrase extractor failed: {e}")
        all_metrics["keyphrase"] = {"error": str(e)}
    
    # 3. Train Topic Modeler
    try:
        metrics = await train_topic_modeler_multilingual(lang_code)
        all_metrics["topic_model"] = metrics
    except Exception as e:
        print(f"[FAIL] Topic modeler failed: {e}")
        all_metrics["topic_model"] = {"error": str(e)}
    
    # 4. Train Relation Extractor
    try:
        metrics = await train_relation_extractor_multilingual(lang_code)
        all_metrics["relation"] = metrics
    except Exception as e:
        print(f"[FAIL] Relation extractor failed: {e}")
        all_metrics["relation"] = {"error": str(e)}
    
    return all_metrics


async def main():
    """Main multilingual training pipeline"""
    print("\n" + "=" * 70)
    print(" VisualVerse - Multilingual NLP Training Pipeline")
    print(" Languages: English | Hindi | Tamil")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # Final Summary
    print("\n" + "=" * 70)
    print(" MULTILINGUAL TRAINING SUMMARY")
    print("=" * 70)
    
    for lang_code, lang_metrics in all_results.items():
        lang_names = {"hi": "Hindi", "ta": "Tamil", "en": "English"}
        lang_name = lang_names.get(lang_code, lang_code)
        
        print(f"\n{lang_name} ({lang_code}):")
        for model_name, metrics in lang_metrics.items():
            if "error" in metrics:
                print(f"  [FAIL] {model_name}: FAILED - {metrics['error']}")
            else:
                acc = metrics.get('accuracy', metrics.get('f1_score', metrics.get('coherence', 'N/A')))
                print(f"  [OK] {model_name}: SUCCESS (score: {acc})")
    
    # Save results
    results_path = Path("training/multilingual_training_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n[FILE] Results saved to {results_path}")
    
    print(f"\n[OK] Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    asyncio.run(main())

