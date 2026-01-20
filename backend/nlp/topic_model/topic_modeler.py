"""
Topic Modeling Module
TRAINABLE model for extracting and clustering topics from text
Used for mind-map hierarchical organization
"""

import os
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import defaultdict
import re


class TopicModeler:
    """
    Trainable Topic Modeling system
    
    Training approach:
    1. Train LDA (Latent Dirichlet Allocation) on document corpus
    2. Train clustering model for grouping related concepts
    3. Learn topic hierarchies from structured data (WikiHow, etc.)
    
    Output:
    - Topic clusters for mind-map organization
    - Hierarchical relationships between topics
    """
    
    MODEL_PATH = "backend/models/topic_model.pkl"
    VECTORIZER_PATH = "backend/models/topic_vectorizer.pkl"
    CLUSTER_PATH = "backend/models/topic_cluster.pkl"
    
    def __init__(self, n_topics: int = 5):
        """Initialize the topic modeler"""
        self._ready = True
        self._trained = False
        self.n_topics = n_topics
        self.lda_model = None
        self.vectorizer = None
        self.cluster_model = None
        self.topic_words = {}
        self.metrics = {}
        
        # Load existing model
        self._load_model()
    
    def is_ready(self) -> bool:
        return self._ready
    
    def is_trained(self) -> bool:
        return self._trained
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics
    
    def _load_model(self):
        """Load pre-trained model"""
        try:
            if os.path.exists(self.MODEL_PATH):
                with open(self.MODEL_PATH, 'rb') as f:
                    saved = pickle.load(f)
                    self.lda_model = saved.get('lda')
                    self.topic_words = saved.get('topic_words', {})
                    self.n_topics = saved.get('n_topics', 5)
                with open(self.VECTORIZER_PATH, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                if os.path.exists(self.CLUSTER_PATH):
                    with open(self.CLUSTER_PATH, 'rb') as f:
                        self.cluster_model = pickle.load(f)
                self._trained = True
        except Exception as e:
            print(f"Could not load topic model: {e}")
    
    def _save_model(self):
        """Save trained model"""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump({
                'lda': self.lda_model,
                'topic_words': self.topic_words,
                'n_topics': self.n_topics
            }, f)
        with open(self.VECTORIZER_PATH, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        if self.cluster_model:
            with open(self.CLUSTER_PATH, 'wb') as f:
                pickle.dump(self.cluster_model, f)
    
    def model_topics(self, preprocessed: Dict[str, Any], 
                     keyphrases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract and cluster topics from text
        
        Uses from preprocessed:
        - lemmas: Base word forms (stopwords removed)
        - sentences: For topic distribution
        - original_text: For context
        
        Returns:
            Dict with:
            - topics: List of topic dicts with words and relevance
            - hierarchy: Hierarchical structure of topics
            - keyphrase_topics: Mapping of keyphrases to topics
        """
        text = preprocessed.get("original_text", "")
        sentences = preprocessed.get("sentences", [])
        
        # Use lemmas from preprocessing (filtered, stopwords removed)
        lemmas = preprocessed.get("lemmas", [])
        
        # If no lemmas available, create from text
        if not lemmas and text:
            # Fallback: simple tokenization
            import re
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            stopwords = {'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                        'have', 'has', 'had', 'do', 'does', 'did', 'and', 'or', 'but'}
            lemmas = [w for w in words if w not in stopwords]
        
        if self._trained and self.lda_model is not None:
            return self._model_with_trained(text, sentences, lemmas, keyphrases)
        
        return self._model_statistical(text, sentences, lemmas, keyphrases)
    
    def _model_with_trained(self, text: str, sentences: List[str], 
                           lemmas: List[str],
                           keyphrases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Model topics using trained LDA with preprocessed lemmas"""
        # Transform text
        if len(sentences) < 2:
            sentences = [text]
        
        doc_term_matrix = self.vectorizer.transform(sentences)
        
        # Get topic distribution
        topic_distributions = self.lda_model.transform(doc_term_matrix)
        
        # Aggregate topic distribution across sentences
        avg_distribution = topic_distributions.mean(axis=0)
        
        # Build topic structure
        topics = []
        for topic_idx in range(self.n_topics):
            topic_data = {
                "id": f"topic_{topic_idx}",
                "relevance": float(avg_distribution[topic_idx]),
                "words": self.topic_words.get(topic_idx, []),
                "label": self._generate_topic_label(self.topic_words.get(topic_idx, []))
            }
            topics.append(topic_data)
        
        # Sort by relevance
        topics.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Assign keyphrases to topics
        keyphrase_topics = self._assign_keyphrases_to_topics(keyphrases, topics, text)
        
        # Build hierarchy
        hierarchy = self._build_hierarchy(topics, keyphrase_topics)
        
        return {
            "topics": topics,
            "hierarchy": hierarchy,
            "keyphrase_topics": keyphrase_topics
        }
    
    def _model_statistical(self, text: str, sentences: List[str], 
                          lemmas: List[str],
                          keyphrases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Statistical topic modeling using preprocessed lemmas
        Uses simple clustering approach when LDA not trained
        """
        # Group keyphrases by co-occurrence
        keyphrase_texts = [kp["phrase"] for kp in keyphrases]
        
        if len(keyphrase_texts) < 2:
            # Single topic case
            return {
                "topics": [{
                    "id": "topic_0",
                    "relevance": 1.0,
                    "words": keyphrase_texts,
                    "label": keyphrase_texts[0] if keyphrase_texts else "Main Topic"
                }],
                "hierarchy": {
                    "root": "Main Topic",
                    "children": [{
                        "label": kp,
                        "children": []
                    } for kp in keyphrase_texts]
                },
                "keyphrase_topics": {kp: "topic_0" for kp in keyphrase_texts}
            }
        
        # Use TF-IDF to create feature vectors for keyphrases
        tfidf = TfidfVectorizer()
        try:
            kp_vectors = tfidf.fit_transform(keyphrase_texts).toarray()
        except:
            # Fallback for very short phrases
            kp_vectors = np.eye(len(keyphrase_texts))
        
        # Determine number of clusters
        n_clusters = min(3, len(keyphrase_texts))
        
        # Cluster keyphrases
        if len(keyphrase_texts) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(kp_vectors)
        else:
            cluster_labels = list(range(len(keyphrase_texts)))
        
        # Build topics from clusters
        cluster_phrases = defaultdict(list)
        for phrase, label in zip(keyphrase_texts, cluster_labels):
            cluster_phrases[label].append(phrase)
        
        topics = []
        keyphrase_topics = {}
        for cluster_id, phrases in cluster_phrases.items():
            topic_id = f"topic_{cluster_id}"
            topics.append({
                "id": topic_id,
                "relevance": len(phrases) / len(keyphrase_texts),
                "words": phrases,
                "label": self._generate_topic_label(phrases)
            })
            for phrase in phrases:
                keyphrase_topics[phrase] = topic_id
        
        # Build hierarchy
        hierarchy = self._build_hierarchy(topics, keyphrase_topics)
        
        return {
            "topics": topics,
            "hierarchy": hierarchy,
            "keyphrase_topics": keyphrase_topics
        }
    
    def _generate_topic_label(self, words: List[str]) -> str:
        """Generate a human-readable label for a topic"""
        if not words:
            return "General"
        
        # Use the most important word(s)
        if len(words) == 1:
            return words[0].title()
        
        # Combine top words
        return " & ".join([w.title() for w in words[:2]])
    
    def _assign_keyphrases_to_topics(self, keyphrases: List[Dict[str, Any]], 
                                     topics: List[Dict[str, Any]], 
                                     text: str) -> Dict[str, str]:
        """Assign keyphrases to their most relevant topics"""
        keyphrase_topics = {}
        
        for kp in keyphrases:
            phrase = kp["phrase"]
            best_topic = topics[0]["id"] if topics else "topic_0"
            best_score = 0
            
            for topic in topics:
                # Check word overlap
                topic_words = set(' '.join(topic["words"]).lower().split())
                phrase_words = set(phrase.lower().split())
                overlap = len(topic_words & phrase_words)
                
                if overlap > best_score:
                    best_score = overlap
                    best_topic = topic["id"]
            
            keyphrase_topics[phrase] = best_topic
        
        return keyphrase_topics
    
    def _build_hierarchy(self, topics: List[Dict[str, Any]], 
                        keyphrase_topics: Dict[str, str]) -> Dict[str, Any]:
        """Build a hierarchical structure for the mind map"""
        # Root is the most relevant topic
        if not topics:
            return {"root": "Main Topic", "children": []}
        
        root_topic = topics[0]
        
        hierarchy = {
            "root": root_topic["label"],
            "root_id": root_topic["id"],
            "children": []
        }
        
        # Add other topics as children
        for topic in topics:
            topic_keyphrases = [
                kp for kp, tid in keyphrase_topics.items() 
                if tid == topic["id"]
            ]
            
            child = {
                "label": topic["label"],
                "id": topic["id"],
                "relevance": topic["relevance"],
                "children": [{"label": kp, "children": []} for kp in topic_keyphrases]
            }
            hierarchy["children"].append(child)
        
        return hierarchy
    
    async def train(self, dataset_name: str = None) -> Dict[str, Any]:
        """
        Train the topic model
        
        Uses:
        - WikiHow articles for hierarchical topic learning
        - BBC News for topic classification
        - Custom document corpus
        
        Training approach:
        1. Build vocabulary from corpus
        2. Train LDA model
        3. Extract topic-word distributions
        4. Optionally train hierarchical clustering
        """
        print("Training topic model...")
        
        # Load training documents
        documents = self._load_training_data(dataset_name)
        
        if not documents:
            return {"error": "No training data available"}
        
        # Create document-term matrix
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            max_features=1000,
            stop_words='english'
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Train LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=20,
            learning_method='online',
            random_state=42,
            n_jobs=-1
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        # Extract topic words
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_words = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_word_indices = topic.argsort()[:-10:-1]
            top_words = [feature_names[i] for i in top_word_indices]
            self.topic_words[topic_idx] = top_words
        
        # Calculate metrics
        perplexity = self.lda_model.perplexity(doc_term_matrix)
        
        # Calculate topic coherence (simplified)
        coherence = self._calculate_coherence(doc_term_matrix, feature_names)
        
        self.metrics = {
            "n_topics": self.n_topics,
            "n_documents": len(documents),
            "vocabulary_size": len(feature_names),
            "perplexity": float(perplexity),
            "coherence": coherence,
            "topic_words": self.topic_words
        }
        
        # Save model
        self._save_model()
        self._trained = True
        
        return self.metrics
    
    def _calculate_coherence(self, doc_term_matrix, feature_names) -> float:
        """Calculate simplified topic coherence score"""
        coherence_scores = []
        
        for topic_idx in range(self.n_topics):
            words = self.topic_words[topic_idx][:5]
            word_indices = [
                list(feature_names).index(w) 
                for w in words if w in feature_names
            ]
            
            if len(word_indices) < 2:
                continue
            
            # Calculate pairwise co-occurrence
            doc_term_dense = doc_term_matrix.toarray()
            pairs_scores = []
            
            for i in range(len(word_indices)):
                for j in range(i + 1, len(word_indices)):
                    wi, wj = word_indices[i], word_indices[j]
                    co_occur = np.sum((doc_term_dense[:, wi] > 0) & (doc_term_dense[:, wj] > 0))
                    individual = np.sum(doc_term_dense[:, wi] > 0)
                    if individual > 0:
                        pairs_scores.append(co_occur / individual)
            
            if pairs_scores:
                coherence_scores.append(np.mean(pairs_scores))
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def _load_training_data(self, dataset_name: str = None) -> List[str]:
        """Load training documents"""
        # Try prepared data first
        data_path = "training/data/topic_data.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                texts = data.get("texts", [])
                # Limit for faster training
                return texts[:500]
        
        # Check for custom dataset
        if dataset_name:
            data_path = f"training/data/{dataset_name}"
            if os.path.exists(data_path):
                with open(data_path, 'rb') as f:
                    return pickle.load(f)
        
        # Use sample documents (WikiHow-style instructional text)
        documents = [
            "How to learn machine learning. Start with basic mathematics including linear algebra and calculus. Learn Python programming. Study statistical concepts. Take online courses. Practice with real datasets. Build projects to apply your knowledge.",
            
            "Understanding climate change. Climate change refers to long-term shifts in temperature. Greenhouse gases trap heat in atmosphere. Human activities increase carbon emissions. Effects include rising sea levels and extreme weather. Solutions involve renewable energy and conservation.",
            
            "Introduction to web development. Web development involves building websites. Frontend uses HTML CSS JavaScript. Backend handles server logic and databases. Frameworks like React and Node simplify development. Deploy applications using cloud services.",
            
            "Basics of healthy nutrition. Balanced diet includes proteins carbohydrates and fats. Vitamins and minerals are essential. Drink adequate water daily. Limit processed foods and sugar. Eat variety of fruits and vegetables.",
            
            "Getting started with photography. Choose appropriate camera equipment. Learn composition rules like rule of thirds. Understand exposure triangle aperture shutter ISO. Practice different lighting conditions. Edit photos using software.",
            
            "Fundamentals of personal finance. Create monthly budget tracking income expenses. Build emergency fund covering months expenses. Invest for long-term growth. Understand compound interest benefits. Diversify investment portfolio.",
            
            "Learning a new language. Immerse yourself in target language. Practice speaking with native speakers. Use spaced repetition for vocabulary. Watch movies and read books. Join language exchange communities.",
            
            "Project management basics. Define project scope and objectives. Create timeline with milestones. Assign tasks to team members. Monitor progress regularly. Adapt to changes and risks.",
            
            "Introduction to meditation. Find quiet comfortable space. Focus on breath awareness. Start short sessions five minutes. Gradually increase duration. Practice consistency daily.",
            
            "Understanding cryptocurrency. Cryptocurrency uses blockchain technology. Bitcoin was first decentralized currency. Mining validates transactions. Wallets store digital assets. Volatility creates investment risk."
        ]
        
        return documents
