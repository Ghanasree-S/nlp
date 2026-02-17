"""
Text Classifier Module
Classifies text as NARRATIVE (story) or INFORMATIONAL (conceptual)
Uses a trained classifier with handcrafted features
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re


class TextClassifier:
    """
    Classifier to determine if text is narrative (story) or informational (conceptual)
    Uses a combination of:
    - Linguistic features (pronouns, verb tenses, etc.)
    - Structural features (sentence patterns, etc.)
    - TF-IDF features
    """
    
    MODEL_PATH = "backend/models/text_classifier.pkl"
    VECTORIZER_PATH = "backend/models/text_vectorizer.pkl"
    
    def __init__(self):
        """Initialize the classifier"""
        self._ready = True
        self._trained = False
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        
        # Try to load existing model
        self._load_model()
    
    def is_ready(self) -> bool:
        """Check if classifier is ready"""
        return self._ready
    
    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self._trained
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return self.metrics
    
    def _load_model(self):
        """Load pre-trained model if exists"""
        try:
            if os.path.exists(self.MODEL_PATH) and os.path.exists(self.VECTORIZER_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                self._trained = True
        except Exception as e:
            print(f"Could not load classifier model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def _extract_features(self, text: str, language: str = "en") -> Dict[str, float]:
        """
        Extract linguistic features for classification (multilingual)
        """
        word_count = len(text.split())
        
        if language == "hi":
            return self._extract_features_hi(text, word_count)
        elif language == "ta":
            return self._extract_features_ta(text, word_count)
        else:
            return self._extract_features_en(text, word_count)
    
    def _extract_features_en(self, text: str, word_count: int) -> Dict[str, float]:
        """English feature extraction"""
        narrative_pronouns = len(re.findall(r'\b(I|he|she|they|we|him|her|them|his|hers|their)\b', text, re.I))
        
        all_ed_words = re.findall(r'\b\w+ed\b', text)
        informational_participles = re.findall(
            r'\b(focused|based|called|used|designed|programmed|improved|defined|'
            r'applied|considered|known|related|described|compared|required|'
            r'automated|powered|specialized|structured|classified|organized|'
            r'developed|advanced|integrated|optimized|processed|generated|'
            r'trained|learned|computed|predicted|clustered|labeled|named|'
            r'proposed|published|implemented|recognized|connected|combined|'
            r'associated|distributed|collected|measured|observed|examined|'
            r'analyzed|derived|achieved|obtained|employed|adopted|constructed)\b',
            text, re.I
        )
        past_tense_verbs = max(0, len(all_ed_words) - len(informational_participles))
        
        dialogue_markers = len(re.findall(r'["\'].*?["\']', text))
        said_verbs = len(re.findall(r'\b(said|asked|replied|whispered|shouted|exclaimed|muttered)\b', text, re.I))
        story_words = len(re.findall(r'\b(once|then|suddenly|finally|afterwards|meanwhile|later)\b', text, re.I))
        
        definition_patterns = len(re.findall(r'\b(is|are|was|were|means|refers|defines|describes)\b', text, re.I))
        bullet_patterns = len(re.findall(r'^\s*[-•*]\s', text, re.M))
        numbered_patterns = len(re.findall(r'^\s*\d+[\.\)]\s', text, re.M))
        technical_patterns = len(re.findall(
            r'\b(according|therefore|however|moreover|furthermore|thus|hence|'
            r'whereas|consequently|specifically|essentially|particularly|'
            r'typically|generally|primarily|significantly|approximately)\b',
            text, re.I
        ))
        explanation_patterns = len(re.findall(
            r'\b(such as|for example|for instance|including|e\.g\.|i\.e\.|'
            r'in other words|that is|refers to|known as|defined as|'
            r'a type of|a form of|a kind of|a subset of|a branch of|'
            r'consists of|involves|enables|allows|provides|facilitates|'
            r'applications like|used for|used in|used to|designed to|'
            r'focused on|based on|capable of|responsible for)\b',
            text, re.I
        ))
        abbreviation_patterns = len(re.findall(r'\([A-Z]{2,}\)', text))
        
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        
        return {
            "narrative_pronouns": narrative_pronouns / max(word_count, 1),
            "past_tense_ratio": past_tense_verbs / max(word_count, 1),
            "dialogue_ratio": dialogue_markers / max(word_count, 1),
            "said_verbs_ratio": said_verbs / max(word_count, 1),
            "story_words_ratio": story_words / max(word_count, 1),
            "definition_ratio": definition_patterns / max(word_count, 1),
            "bullet_patterns": bullet_patterns,
            "numbered_patterns": numbered_patterns,
            "technical_words_ratio": technical_patterns / max(word_count, 1),
            "explanation_patterns": explanation_patterns / max(word_count, 1),
            "abbreviation_patterns": abbreviation_patterns / max(word_count, 1),
            "avg_sentence_length": avg_sentence_length,
            "word_count": word_count
        }
    
    def _extract_features_hi(self, text: str, word_count: int) -> Dict[str, float]:
        """Hindi feature extraction"""
        narrative_pronouns = len(re.findall(
            r'(\u092e\u0948\u0902|\u0939\u092e|\u0924\u0941\u092e|\u0935\u0939|\u0935\u0947|\u0909\u0938\u0928\u0947|\u0909\u0938\u0915\u093e|\u0909\u0938\u0915\u0940|\u0909\u0928\u094d\u0939\u094b\u0902\u0928\u0947)', text
        ))  # मैं, हम, तुम, वह, वे, उसने, उसका, उसकी, उन्होंने
        
        past_tense_verbs = len(re.findall(
            r'(\u0925\u093e|\u0925\u0940|\u0925\u0947|\u0917\u092f\u093e|\u0917\u0908|\u0917\u090f|\u0906\u092f\u093e|\u0906\u0908|\u0906\u090f|\u0932\u093f\u092f\u093e|\u0926\u093f\u092f\u093e|\u0915\u093f\u092f\u093e)', text
        ))  # था, थी, थे, गया, गई, गए, आया, आई, आए, लिया, दिया, किया
        
        story_words = len(re.findall(
            r'(\u090f\u0915 \u092c\u093e\u0930|\u092b\u093f\u0930|\u0905\u091a\u093e\u0928\u0915|\u0924\u092d\u0940|\u0906\u0916\u093f\u0930\u0915\u093e\u0930|\u092a\u0939\u0932\u0947|\u092c\u093e\u0926 \u092e\u0947\u0902)', text
        ))  # एक बार, फिर, अचानक, तभी, आखिरकार, पहले, बाद में
        
        dialogue_markers = len(re.findall(r'["\u201C\u201D].*?["\u201C\u201D]', text))
        said_verbs = len(re.findall(
            r'(\u092c\u094b\u0932\u093e|\u0915\u0939\u093e|\u092a\u0942\u091b\u093e|\u091a\u093f\u0932\u094d\u0932\u093e\u092f\u093e|\u0938\u0941\u0928\u093e\u092f\u093e)', text
        ))  # बोला, कहा, पूछा, चिल्लाया, सुनाया
        
        definition_patterns = len(re.findall(
            r'(\u0939\u0948|\u0939\u0948\u0902|\u0939\u094b\u0924\u093e \u0939\u0948|\u0915\u0939\u0924\u0947 \u0939\u0948\u0902|\u0906\u0927\u093e\u0930\u093f\u0924)', text
        ))  # है, हैं, होता है, कहते हैं, आधारित
        
        technical_patterns = len(re.findall(
            r'(\u0907\u0938\u0932\u093f\u090f|\u0909\u0926\u093e\u0939\u0930\u0923|\u0935\u093f\u0936\u0947\u0937|\u0938\u093e\u092e\u093e\u0928\u094d\u092f\u0924\u0903|\u092e\u0941\u0916\u094d\u092f)', text
        ))  # इसलिए, उदाहरण, विशेष, सामान्यतः, मुख्य
        
        sentences = re.split(r'[\u0964\.]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        
        return {
            "narrative_pronouns": narrative_pronouns / max(word_count, 1),
            "past_tense_ratio": past_tense_verbs / max(word_count, 1),
            "dialogue_ratio": dialogue_markers / max(word_count, 1),
            "said_verbs_ratio": said_verbs / max(word_count, 1),
            "story_words_ratio": story_words / max(word_count, 1),
            "definition_ratio": definition_patterns / max(word_count, 1),
            "bullet_patterns": len(re.findall(r'^\s*[-\u2022*]\s', text, re.M)),
            "numbered_patterns": len(re.findall(r'^\s*\d+[\.\)]\s', text, re.M)),
            "technical_words_ratio": technical_patterns / max(word_count, 1),
            "explanation_patterns": 0,
            "abbreviation_patterns": len(re.findall(r'\([A-Z]{2,}\)', text)) / max(word_count, 1),
            "avg_sentence_length": avg_sentence_length,
            "word_count": word_count
        }
    
    def _extract_features_ta(self, text: str, word_count: int) -> Dict[str, float]:
        """Tamil feature extraction"""
        narrative_pronouns = len(re.findall(
            r'(\u0BA8\u0BBE\u0BA9\u0BCD|\u0BA8\u0BBE\u0BAE\u0BCD|\u0BA8\u0BC0|\u0B85\u0BB5\u0BA9\u0BCD|\u0B85\u0BB5\u0BB3\u0BCD|\u0B85\u0BB5\u0BB0\u0BCD|\u0B85\u0BB5\u0BB0\u0BCD\u0B95\u0BB3\u0BCD)', text
        ))  # நான், நாம், நீ, அவன், அவள், அவர், அவர்கள்
        
        past_tense_verbs = len(re.findall(
            r'(\u0BA9\u0BBE\u0BA9\u0BCD|\u0BA9\u0BBE\u0BB3\u0BCD|\u0BA9\u0BBE\u0BB0\u0BCD|\u0B9A\u0BC6\u0BA9\u0BCD\u0BB1\u0BBE\u0BA9\u0BCD|\u0BB5\u0BA8\u0BCD\u0BA4\u0BBE\u0BA9\u0BCD|\u0B87\u0BB0\u0BC1\u0BA8\u0BCD\u0BA4\u0BA4\u0BC1)', text
        ))  # னான், னாள், னார், சென்றான், வந்தான், இருந்தது
        
        story_words = len(re.findall(
            r'(\u0B92\u0BB0\u0BC1 \u0BA8\u0BBE\u0BB3\u0BCD|\u0BA4\u0BBF\u0B9F\u0BC0\u0BB0\u0BC6\u0BA9\u0BCD|\u0B85\u0BAA\u0BCD\u0BAA\u0BCB\u0BA4\u0BC1|\u0B87\u0BB1\u0BC1\u0BA4\u0BBF\u0BAF\u0BBF\u0BB2\u0BCD|\u0BAA\u0BBF\u0BA9\u0BCD\u0BA9\u0BB0\u0BCD)', text
        ))  # ஒரு நாள், திடீரென், அப்போது, இறுதியில், பின்னர்
        
        dialogue_markers = len(re.findall(r'["\u201C\u201D].*?["\u201C\u201D]', text))
        said_verbs = len(re.findall(
            r'(\u0B9A\u0BCA\u0BA9\u0BCD\u0BA9\u0BBE\u0BB0\u0BCD|\u0B95\u0BC7\u0B9F\u0BCD\u0B9F\u0BBE\u0BB0\u0BCD|\u0B95\u0BC2\u0BB1\u0BBF\u0BA9\u0BBE\u0BB0\u0BCD)', text
        ))  # சொன்னார், கேட்டார், கூறினார்
        
        definition_patterns = len(re.findall(
            r'(\u0B86\u0B95\u0BC1\u0BAE\u0BCD|\u0B8E\u0BA9\u0BCD\u0BAA\u0BA4\u0BC1|\u0B89\u0BB3\u0BCD\u0BB3\u0BA4\u0BC1)', text
        ))  # ஆகும், என்பது, உள்ளது
        
        technical_patterns = len(re.findall(
            r'(\u0B8E\u0BA9\u0BB5\u0BC7|\u0B89\u0BA4\u0BBE\u0BB0\u0BA3\u0BAE\u0BBE\u0B95|\u0BAA\u0BCA\u0BA4\u0BC1\u0BB5\u0BBE\u0B95|\u0BAE\u0BC1\u0B95\u0BCD\u0B95\u0BBF\u0BAF\u0BAE\u0BBE\u0B95)', text
        ))  # எனவே, உதாரணமாக, பொதுவாக, முக்கியமாக
        
        sentences = re.split(r'[\.\u0964]+', text)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0
        
        return {
            "narrative_pronouns": narrative_pronouns / max(word_count, 1),
            "past_tense_ratio": past_tense_verbs / max(word_count, 1),
            "dialogue_ratio": dialogue_markers / max(word_count, 1),
            "said_verbs_ratio": said_verbs / max(word_count, 1),
            "story_words_ratio": story_words / max(word_count, 1),
            "definition_ratio": definition_patterns / max(word_count, 1),
            "bullet_patterns": len(re.findall(r'^\s*[-\u2022*]\s', text, re.M)),
            "numbered_patterns": len(re.findall(r'^\s*\d+[\.\)]\s', text, re.M)),
            "technical_words_ratio": technical_patterns / max(word_count, 1),
            "explanation_patterns": 0,
            "abbreviation_patterns": len(re.findall(r'\([A-Z]{2,}\)', text)) / max(word_count, 1),
            "avg_sentence_length": avg_sentence_length,
            "word_count": word_count
        }
    
    def classify(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify text as narrative or informational (multilingual)
        
        Args:
            preprocessed: Output from TextPreprocessor.process()
        
        Returns:
            Dict with type, confidence, features, and language
        """
        text = preprocessed.get("original_text", preprocessed.get("cleaned_text", ""))
        language = preprocessed.get("language", "en")
        
        # If model is trained, use it (English model)
        if self._trained and self.model is not None and language == "en":
            return self._classify_with_model(text)
        
        # Otherwise use rule-based classification (works for all languages)
        return self._classify_rule_based(text, preprocessed, language)
    
    def _classify_with_model(self, text: str) -> Dict[str, Any]:
        """Classify using trained model"""
        features = self._extract_features(text)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Get TF-IDF features
        tfidf_features = self.vectorizer.transform([text]).toarray()
        
        # Combine features
        combined_features = np.hstack([feature_vector, tfidf_features])
        
        # Predict
        prediction = self.model.predict(combined_features)[0]
        probabilities = self.model.predict_proba(combined_features)[0]
        
        text_type = "narrative" if prediction == 1 else "informational"
        confidence = max(probabilities)
        
        return {
            "type": text_type,
            "confidence": float(confidence),
            "features": features
        }
    
    def _classify_rule_based(self, text: str, preprocessed: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
        """
        Rule-based classification when model is not trained
        Uses heuristics based on linguistic features (multilingual)
        """
        features = self._extract_features(text, language)
        
        # Calculate narrative score
        narrative_score = (
            features["narrative_pronouns"] * 15 +
            features["past_tense_ratio"] * 10 +
            features["dialogue_ratio"] * 20 +
            features["said_verbs_ratio"] * 25 +
            features["story_words_ratio"] * 15
        )
        
        # Calculate informational score
        informational_score = (
            features["definition_ratio"] * 15 +
            features["bullet_patterns"] * 5 +
            features["numbered_patterns"] * 5 +
            features["technical_words_ratio"] * 20 +
            features["explanation_patterns"] * 25 +
            features["abbreviation_patterns"] * 15
        )
        
        # Check for characters (strong narrative indicator)
        if len(preprocessed.get("characters", [])) > 0:
            narrative_score += 0.2
        
        # Determine type
        if narrative_score > informational_score:
            text_type = "narrative"
            confidence = min(0.95, 0.5 + (narrative_score - informational_score))
        else:
            text_type = "informational"
            confidence = min(0.95, 0.5 + (informational_score - narrative_score))
        
        return {
            "type": text_type,
            "confidence": confidence,
            "features": features,
            "narrative_score": narrative_score,
            "informational_score": informational_score
        }
    
    async def train(self, dataset_path: str = None) -> Dict[str, Any]:
        """
        Train the classifier on labeled data
        
        Dataset format: List of (text, label) where label is 0 (informational) or 1 (narrative)
        """
        # Load or create training data
        if dataset_path and os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
            texts = [item[0] for item in data]
            labels = [item[1] for item in data]
        else:
            # Use sample training data
            texts, labels = self._get_sample_training_data()
        
        # Extract features
        feature_vectors = []
        for text in texts:
            features = self._extract_features(text)
            feature_vectors.append(list(features.values()))
        
        feature_vectors = np.array(feature_vectors)
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        
        # Combine features
        combined_features = np.hstack([feature_vectors, tfidf_features])
        labels = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            combined_features, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.metrics = {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Save model
        self._save_model()
        self._trained = True
        
        return self.metrics
    
    def _get_sample_training_data(self) -> Tuple[List[str], List[int]]:
        """
        Returns sample training data for the classifier
        Label: 0 = informational, 1 = narrative
        """
        narratives = [
            "Once upon a time, there was a young girl named Alice who fell down a rabbit hole. She found herself in a magical world full of strange creatures.",
            "John walked into the room and saw Mary sitting by the window. 'What are you doing here?' he asked. She turned and smiled at him.",
            "The knight drew his sword and faced the dragon. Fire erupted from the beast's mouth, but he dodged just in time.",
            "Sarah had been waiting for this moment her entire life. As she stepped onto the stage, the crowd erupted in applause.",
            "He ran through the forest, his heart pounding. Behind him, he could hear the wolves getting closer.",
            "The old man sat on the porch, watching the sunset. He remembered the day he first met his wife, sixty years ago.",
            "Detective Miller examined the crime scene carefully. Something didn't add up. The victim knew their killer.",
            "The spaceship landed on the alien planet. Captain Chen stepped out first, her eyes widening at the strange landscape.",
            "Little Tommy found a mysterious key in his grandmother's attic. He wondered what secrets it could unlock.",
            "The princess escaped from the tower using a rope made of her own hair. She ran through the dark forest until dawn."
        ]
        
        informational = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data. There are three main types: supervised, unsupervised, and reinforcement learning.",
            "The water cycle consists of evaporation, condensation, and precipitation. Water evaporates from oceans and lakes, forms clouds, and falls as rain.",
            "Python is a high-level programming language known for its simplicity. Key features include: dynamic typing, automatic memory management, and extensive libraries.",
            "Climate change refers to long-term shifts in global temperatures. The main causes are greenhouse gas emissions, deforestation, and industrial activities.",
            "The human heart has four chambers: two atria and two ventricles. It pumps blood through the circulatory system continuously.",
            "Photosynthesis is the process by which plants convert sunlight into energy. The chemical equation is: 6CO2 + 6H2O → C6H12O6 + 6O2.",
            "Democracy is a system of government where power is held by the people. Key principles include voting rights, freedom of speech, and rule of law.",
            "Neural networks are computing systems inspired by biological brains. They consist of input layers, hidden layers, and output layers.",
            "The Renaissance was a cultural movement from the 14th to 17th century. It originated in Italy and spread throughout Europe.",
            "Blockchain is a distributed ledger technology. It provides transparency, security, and immutability for digital transactions."
        ]
        
        texts = narratives + informational
        labels = [1] * len(narratives) + [0] * len(informational)
        
        return texts, labels
