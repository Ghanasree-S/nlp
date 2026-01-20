"""
Hybrid NLP + ML Keyphrase Extraction Model

NLP Techniques Used (for Candidate Generation):
1. Named Entity Recognition (NER) - Extract PERSON, ORG, GPE entities
2. Noun Chunks - Extract compound noun phrases
3. Dependency Parsing - Extract subjects and objects (nsubj, dobj, pobj)
4. POS Tagging - Filter for nouns (NOUN, PROPN)

ML Technique Used (for Scoring/Ranking):
- Gradient Boosting Classifier with 8 features
"""

import os
import pickle
import re
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# NLP imports
try:
    import spacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False


class KeyphraseExtractor:
    """
    Hybrid NLP + ML Keyphrase Extraction
    
    Pipeline:
    1. NLP Candidate Generation (SpaCy):
       - NER: Extract named entities (ORG, PRODUCT, GPE)
       - Noun Chunks: Extract compound nouns
       - Dependencies: Extract subjects/objects
       - POS Tags: Filter for nouns
    
    2. ML Scoring (Gradient Boosting):
       - 8 features: position, frequency, length, etc.
       - Trained on academic abstracts
       - Predicts P(candidate is keyphrase)
    """
    
    MODEL_PATH = "backend/models/keyphrase_model.pkl"
    TFIDF_PATH = "backend/models/keyphrase_tfidf.pkl"
    
    STOPWORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'to', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'and',
        'but', 'or', 'this', 'that', 'these', 'those', 'it', 'its',
        # Additional stopwords for cleaner mindmaps
        'also', 'just', 'only', 'even', 'such', 'very', 'too', 'much', 'many',
        'more', 'most', 'other', 'some', 'any', 'all', 'each', 'every', 'both',
        'few', 'than', 'them', 'they', 'their', 'them', 'we', 'you', 'our',
        'what', 'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
        'there', 'here', 'then', 'now', 'out', 'up', 'down', 'over', 'under',
        'about', 'so', 'because', 'if', 'while', 'although', 'though', 'until',
        'unless', 'since', 'whether', 'either', 'neither', 'not', 'no', 'yes',
        'yet', 'still', 'already', 'always', 'never', 'often', 'sometimes',
        'fuelled', 'causing', 'cut', 'clear', 'rates', 'profit', 'alarming'
    }
    
    # Common verb forms to filter out
    VERBS = {
        'adds', 'consists', 'enables', 'structures', 'styles', 'stores', 'uses',
        'requires', 'provides', 'includes', 'contains', 'creates', 'makes',
        'gives', 'takes', 'gets', 'sets', 'puts', 'runs', 'shows', 'finds',
        'keeps', 'lets', 'helps', 'allows', 'needs', 'wants', 'starts', 'ends',
        'goes', 'comes', 'brings', 'becomes', 'remains', 'seems', 'appears',
        'happens', 'occurs', 'causes', 'leads', 'results', 'affects', 'impacts'
    }
    
    def __init__(self):
        self.model = None
        self.tfidf = None
        self._trained = False
        self._ready = True
        self.metrics = {}
        self._load_model()
    
    def is_ready(self) -> bool:
        return self._ready
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def _load_model(self):
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.TFIDF_PATH, 'rb') as f:
                    self.tfidf = pickle.load(f)
                self._trained = True
        except:
            pass
    
    def _save_model(self):
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.TFIDF_PATH, 'wb') as f:
            pickle.dump(self.tfidf, f)
    
    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()
    
    def _generate_candidates_nlp(self, text: str, doc) -> Dict[str, Dict]:
        """
        NLP-based Candidate Generation using SpaCy
        
        Extracts candidates using 4 NLP techniques:
        1. NER - Named Entity Recognition
        2. Noun Chunks - Compound noun phrases
        3. Dependencies - Subjects and objects
        4. POS Tags - NOUN and PROPN tokens
        
        Returns dict with candidate -> metadata (source, pos, is_entity, etc.)
        """
        candidates = {}
        
        # 1. NAMED ENTITIES (NER) - Unit 4
        # Extract ORG, PRODUCT, GPE, WORK_OF_ART, LAW - likely keyphrases
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW', 'EVENT']:
                phrase = ent.text.strip()
                if len(phrase) > 2 and phrase.lower() not in self.STOPWORDS:
                    candidates[phrase.lower()] = {
                        'original': phrase,
                        'source': 'NER',
                        'entity_type': ent.label_,
                        'is_entity': 1,
                        'is_noun_chunk': 0,
                        'is_subject': 0,
                        'pos': 'ENTITY'
                    }
        
        # 2. NOUN CHUNKS - Compound noun phrases - Unit 1 (Syntax)
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            phrase_lower = phrase.lower()
            # Filter: remove if starts/ends with stopword
            words = phrase_lower.split()
            if words and words[0] not in self.STOPWORDS and words[-1] not in self.STOPWORDS:
                if len(phrase) > 3 and phrase_lower not in self.STOPWORDS:
                    if phrase_lower not in candidates:
                        candidates[phrase_lower] = {
                            'original': phrase,
                            'source': 'NOUN_CHUNK',
                            'entity_type': None,
                            'is_entity': 0,
                            'is_noun_chunk': 1,
                            'is_subject': 0,
                            'pos': chunk.root.pos_
                        }
                    else:
                        candidates[phrase_lower]['is_noun_chunk'] = 1
        
        # 3. DEPENDENCY PARSING - Subjects and Objects - Unit 1 (Syntax)
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr']:
                phrase = token.text.strip()
                phrase_lower = phrase.lower()
                if len(phrase) > 3 and phrase_lower not in self.STOPWORDS:
                    if phrase_lower not in candidates:
                        candidates[phrase_lower] = {
                            'original': phrase,
                            'source': 'DEPENDENCY',
                            'entity_type': None,
                            'is_entity': 0,
                            'is_noun_chunk': 0,
                            'is_subject': 1 if 'subj' in token.dep_ else 0,
                            'pos': token.pos_
                        }
                    else:
                        if 'subj' in token.dep_:
                            candidates[phrase_lower]['is_subject'] = 1
        
        # 4. POS TAGGING - NOUN and PROPN tokens - Unit 4
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop:
                phrase = token.text.strip()
                phrase_lower = phrase.lower()
                if len(phrase) > 4 and phrase_lower not in self.STOPWORDS:
                    if phrase_lower not in candidates:
                        candidates[phrase_lower] = {
                            'original': phrase,
                            'source': 'POS',
                            'entity_type': None,
                            'is_entity': 0,
                            'is_noun_chunk': 0,
                            'is_subject': 0,
                            'pos': token.pos_
                        }
        
        return candidates
    
    def _generate_candidates(self, text: str, max_ngram: int = 3) -> List[str]:
        """
        Hybrid Candidate Generation:
        - If SpaCy available: Use NLP (NER, POS, Chunks, Dependencies)
        - Fallback: Use regex patterns
        """
        # Try NLP-based extraction first
        if NLP_AVAILABLE:
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(text)
                nlp_candidates = self._generate_candidates_nlp(text, doc)
                
                # Store metadata for feature extraction later
                self._candidate_metadata = nlp_candidates
                
                # Return just the phrases
                return list(nlp_candidates.keys())[:50]
            except:
                pass  # Fall through to regex
        
        # Fallback: Regex-based extraction
        candidates = set()
        original_words = text.split()
        tokens = self._tokenize(text)
        text_lower = text.lower()
        
        # Hyphenated terms
        hyphenated = re.findall(r'\b[a-zA-Z]+-[a-zA-Z]+(?:-[a-zA-Z]+)?\b', text)
        for term in hyphenated:
            if len(term) > 4:
                candidates.add(term.lower())
        
        # N-grams (2-3 words)
        for i in range(len(tokens)):
            for ngram_len in [2, 3]:
                if i + ngram_len <= len(tokens):
                    ngram = tokens[i:i+ngram_len]
                    if all(w not in self.STOPWORDS and len(w) > 3 for w in ngram):
                        if not any(w in self.VERBS for w in ngram):
                            candidates.add(' '.join(ngram))
        
        # Single nouns
        for token in tokens:
            if token not in self.STOPWORDS and token not in self.VERBS and len(token) > 4:
                candidates.add(token)
        
        # Cleanup
        cleaned = set()
        for c in candidates:
            words = c.split()
            if words[0] in self.STOPWORDS or words[-1] in self.STOPWORDS:
                continue
            if any(w in self.VERBS for w in words):
                continue
            if len(c) < 4:
                continue
            cleaned.add(c)
        
        self._candidate_metadata = {}  # Empty for regex fallback
        return list(cleaned)[:50]
    
    def _is_keyphrase_match(self, candidate: str, true_keyphrases: set) -> bool:
        """
        IMPROVED matching - fuzzy match
        Checks if candidate matches any true keyphrase
        """
        candidate = candidate.lower().strip()
        
        for kp in true_keyphrases:
            kp = kp.lower().strip()
            
            # Exact match
            if candidate == kp:
                return True
            
            # Candidate is part of keyphrase
            if candidate in kp:
                return True
            
            # Keyphrase is part of candidate
            if kp in candidate:
                return True
            
            # Word overlap > 50%
            cand_words = set(candidate.split())
            kp_words = set(kp.split())
            if cand_words and kp_words:
                overlap = len(cand_words & kp_words)
                if overlap / max(len(cand_words), len(kp_words)) >= 0.5:
                    return True
        
        return False
    
    def _extract_features(self, candidate: str, text: str, text_len: int) -> np.ndarray:
        """
        Extract features for Gradient Boosting scorer
        
        Features (11 total):
        - 8 Statistical: position, frequency, length, spread, etc.
        - 3 NLP-derived: is_entity, is_noun_chunk, is_subject (from SpaCy)
        """
        text_lower = text.lower()
        candidate_lower = candidate.lower()
        
        # === STATISTICAL FEATURES (8) ===
        
        # Position features
        pos = text_lower.find(candidate_lower)
        position = 1 - (pos / text_len) if pos >= 0 else 0
        
        # Frequency
        freq = text_lower.count(candidate_lower)
        freq_norm = min(freq / 5.0, 1.0)
        
        # Length features
        word_count = len(candidate.split())
        word_count_norm = word_count / 3.0
        char_len = len(candidate) / 30.0
        
        # Position features
        in_first_100 = 1.0 if candidate_lower in text_lower[:100] else 0.0
        in_first_200 = 1.0 if candidate_lower in text_lower[:200] else 0.0
        
        # Spread (last - first occurrence)
        last_pos = text_lower.rfind(candidate_lower)
        spread = (last_pos - pos) / text_len if last_pos > pos else 0
        
        # Capitalization in original
        has_caps = 1.0 if any(c.isupper() for c in text[max(0,pos):pos+len(candidate)] if pos >= 0) else 0.0
        
        # === NLP-DERIVED FEATURES (3) ===
        # These come from SpaCy: NER, Noun Chunks, Dependency Parsing
        
        is_entity = 0.0
        is_noun_chunk = 0.0
        is_subject = 0.0
        
        if hasattr(self, '_candidate_metadata') and candidate_lower in self._candidate_metadata:
            meta = self._candidate_metadata[candidate_lower]
            is_entity = float(meta.get('is_entity', 0))
            is_noun_chunk = float(meta.get('is_noun_chunk', 0))
            is_subject = float(meta.get('is_subject', 0))
        
        return np.array([
            # Statistical features (8)
            position, freq_norm, word_count_norm, char_len,
            in_first_100, in_first_200, spread, has_caps,
            # NLP-derived features (3)
            is_entity, is_noun_chunk, is_subject
        ])
    
    def extract(self, preprocessed: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Extract keyphrases using PREPROCESSED DATA from the preprocessing stage
        
        Uses from preprocessed:
        - entities: NER results (ORG, PRODUCT, GPE)
        - noun_chunks: Compound noun phrases
        - nouns: NOUN/PROPN tokens (filtered by POS)
        - subjects: Words with nsubj dependency
        - original_text: For position/frequency features
        """
        text = preprocessed.get("original_text", "")
        
        # Generate candidates FROM PREPROCESSED DATA
        candidates, metadata = self._generate_candidates_from_preprocessed(preprocessed)
        
        if not candidates:
            return []
        
        # Store metadata for feature extraction
        self._candidate_metadata = metadata
        
        if self._trained and self.model is not None:
            return self._extract_with_model(text, candidates, top_k)
        return self._extract_statistical(text, candidates, top_k)
    
    def _generate_candidates_from_preprocessed(self, preprocessed: Dict[str, Any]) -> Tuple[List[str], Dict]:
        """
        Generate candidates using PREPROCESSED DATA
        
        This properly uses the NLP outputs from preprocessing:
        1. Entities (from NER)
        2. Noun Chunks (from SpaCy)
        3. Nouns (from POS tagging)
        4. Subjects (from dependency parsing)
        """
        candidates = {}
        text = preprocessed.get("original_text", "").lower()
        
        # 1. FROM NER - Named Entities (ORG, PRODUCT, GPE, etc.)
        for entity in preprocessed.get("entities", []):
            phrase = entity.get("text", "").strip()
            label = entity.get("label", "")
            if len(phrase) > 2 and phrase.lower() not in self.STOPWORDS:
                if label in ['ORG', 'PRODUCT', 'GPE', 'WORK_OF_ART', 'LAW', 'EVENT']:
                    candidates[phrase.lower()] = {
                        'original': phrase,
                        'source': 'NER',
                        'is_entity': 1,
                        'is_noun_chunk': 0,
                        'is_subject': 0,
                        'entity_type': label
                    }
        
        # 2. FROM NOUN CHUNKS - Compound nouns
        for chunk in preprocessed.get("noun_chunks", []):
            phrase = chunk.get("text", "").strip() if isinstance(chunk, dict) else str(chunk).strip()
            phrase_lower = phrase.lower()
            words = phrase_lower.split()
            
            # Filter: remove if starts/ends with stopword
            if words and words[0] not in self.STOPWORDS and words[-1] not in self.STOPWORDS:
                if len(phrase) > 3 and phrase_lower not in self.STOPWORDS:
                    if phrase_lower not in candidates:
                        candidates[phrase_lower] = {
                            'original': phrase,
                            'source': 'NOUN_CHUNK',
                            'is_entity': 0,
                            'is_noun_chunk': 1,
                            'is_subject': 0,
                            'entity_type': None
                        }
                    else:
                        candidates[phrase_lower]['is_noun_chunk'] = 1
        
        # 3. FROM POS - Nouns (NOUN, PROPN)
        for noun in preprocessed.get("nouns", []):
            phrase = noun.get("text", "").strip() if isinstance(noun, dict) else str(noun).strip()
            phrase_lower = phrase.lower()
            
            if len(phrase) > 3 and phrase_lower not in self.STOPWORDS:
                if phrase_lower not in candidates:
                    candidates[phrase_lower] = {
                        'original': phrase,
                        'source': 'POS',
                        'is_entity': 0,
                        'is_noun_chunk': 0,
                        'is_subject': 0,
                        'entity_type': None
                    }
        
        # 4. FROM DEPENDENCIES - Subjects (nsubj)
        for subject in preprocessed.get("subjects", []):
            subj_lower = subject.lower() if isinstance(subject, str) else str(subject).lower()
            if subj_lower in candidates:
                candidates[subj_lower]['is_subject'] = 1
        
        return list(candidates.keys())[:50], candidates
    
    def _extract_with_model(self, text: str, candidates: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Score candidates using Gradient Boosting model"""
        if not candidates:
            return []
        
        text_len = max(len(text), 1)
        features = [self._extract_features(c, text, text_len) for c in candidates]
        X = np.array(features)
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = self.model.predict(X)
        
        scored = sorted(zip(candidates, probs), key=lambda x: x[1], reverse=True)
        return [{"phrase": p, "score": float(s), "type": "concept"} for p, s in scored[:top_k]]
    
    def _extract_statistical(self, text: str, candidates: List[str], top_k: int) -> List[Dict[str, Any]]:
        """Statistical fallback scoring when ML model not trained"""
        text_lower = text.lower()
        
        scored = []
        for c in candidates:
            freq = text_lower.count(c.lower())
            pos = 1 - (text_lower.find(c.lower()) / max(len(text_lower), 1))
            word_count = len(c.split())
            
            # Heavy boost for technical terms (capitalized, acronyms, special chars)
            is_technical = any(char.isupper() for char in c) or '-' in c or '.' in c
            technical_boost = 1.5 if is_technical else 0
            
            # Prefer single words over multi-word phrases
            single_word_boost = 0.8 if word_count == 1 else -0.3 * (word_count - 1)
            
            # Boost for terms in first sentence
            first_sentence = text_lower.split('.')[0] if '.' in text_lower else text_lower
            in_first = 0.3 if c.lower() in first_sentence else 0
            
            # Penalize common generic words
            generic_words = ['data', 'application', 'system', 'technology', 'information']
            is_generic = -0.5 if c.lower() in generic_words and word_count == 1 else 0
            
            score = freq * 0.3 + pos * 0.15 + technical_boost + single_word_boost + in_first + is_generic
            scored.append((c, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return clean, deduplicated results
        seen = set()
        unique_results = []
        for phrase, score in scored:
            phrase_lower = phrase.lower()
            if phrase_lower not in seen:
                seen.add(phrase_lower)
                unique_results.append({"phrase": phrase, "score": float(score), "type": "concept"})
                if len(unique_results) >= top_k * 2:
                    break
        
        return unique_results[:top_k * 2]
    
    async def train(self, dataset_name: str = None) -> Dict[str, Any]:
        """Train with ALL data and improved matching"""
        print("Training Keyphrase Extractor (IMPROVED)...")
        
        texts, keyphrases_list = self._load_training_data(dataset_name)
        
        # USE MORE DATA
        max_docs = 1000  # Increased from 300
        texts = texts[:max_docs]
        keyphrases_list = keyphrases_list[:max_docs]
        
        print(f"   Using {len(texts)} documents")
        
        # Process in batches
        batch_size = 200
        all_X = []
        all_y = []
        
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            print(f"   Processing docs {batch_start}-{batch_end}...")
            
            for i in range(batch_start, batch_end):
                text = texts[i]
                true_kps = set(keyphrases_list[i])
                text_len = max(len(text), 1)
                
                candidates = self._generate_candidates(text)
                
                for candidate in candidates:
                    features = self._extract_features(candidate, text, text_len)
                    all_X.append(features)
                    
                    # IMPROVED MATCHING
                    is_kp = 1 if self._is_keyphrase_match(candidate, true_kps) else 0
                    all_y.append(is_kp)
        
        X = np.array(all_X)
        y = np.array(all_y)
        
        print(f"   Total examples: {len(X)}")
        print(f"   Positive: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Use Gradient Boosting (better for imbalanced)
        print("   Training Gradient Boosting classifier...")
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary', zero_division=0
        )
        
        self.metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "num_documents": len(texts),
            "num_examples": len(X),
            "positive_rate": float(sum(y) / len(y))
        }
        
        self._save_model()
        self._trained = True
        
        print(f"\n   ✅ Training complete!")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        return self.metrics
    
    def _load_training_data(self, dataset_name: str = None) -> Tuple[List[str], List[List[str]]]:
        data_path = "training/data/keyphrase_data.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            return data["texts"], data["keyphrases"]
        return (["Machine learning is AI."], [["machine learning", "ai"]])
