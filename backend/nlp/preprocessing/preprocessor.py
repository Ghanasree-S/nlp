"""
Text Preprocessor Module
Creates a rich SpaCy Doc object for the entire NLP pipeline

OUTPUT: Preprocessed dict containing:
- tokens: List of token dicts with text, lemma, pos, dep
- entities: List of NER entities (ORG, PERSON, GPE, etc.)
- noun_chunks: Compound noun phrases
- dependencies: Dependency parse information
- sentences: Sentence splits
"""

import re
from typing import Dict, List, Any

# Try SpaCy first, fallback to NLTK
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk import pos_tag, ne_chunk
    from nltk.chunk import tree2conlltags
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """
    Unified Text Preprocessing Pipeline
    
    Uses SpaCy (preferred) or NLTK (fallback) to extract:
    - Tokens with POS tags
    - Named Entities (NER)
    - Noun Chunks
    - Dependency Parse
    - Lemmas
    
    This output is passed to ALL downstream models:
    - Classifier (uses tokens for linguistic features)
    - Keyphrase Extractor (uses POS, NER, noun_chunks)
    - Topic Modeler (uses lemmas)
    - Relation Extractor (uses dependencies)
    """
    
    def __init__(self):
        """Initialize the preprocessor with SpaCy or NLTK"""
        self._ready = False
        self.nlp = None
        self.use_spacy = False
        
        # Try SpaCy first (preferred for dependencies)
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.use_spacy = True
                self._ready = True
                print("✓ Preprocessor: Using SpaCy (full NLP)")
            except OSError:
                print("⚠ SpaCy model not found, trying NLTK...")
        
        # Fallback to NLTK
        if not self.use_spacy and NLTK_AVAILABLE:
            try:
                nltk.data.find('tokenizers/punkt')
                self.stop_words = set(stopwords.words('english'))
                self._ready = True
                print("✓ Preprocessor: Using NLTK (limited NLP)")
            except:
                self._download_nltk_data()
                self.stop_words = set(stopwords.words('english'))
                self._ready = True
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 
                     'maxent_ne_chunker', 'words']
        for name in resources:
            try:
                nltk.download(name, quiet=True)
            except:
                pass
    
    def is_ready(self) -> bool:
        return self._ready
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Main preprocessing function
        
        Returns a rich dict that ALL downstream models use:
        - tokens: [{text, lemma, pos, dep, is_stop}]
        - entities: [{text, label, start, end}]
        - noun_chunks: [str]
        - dependencies: [{head, dep, child}]
        - lemmas: [str] (for topic modeling)
        """
        if self.use_spacy:
            return self._process_spacy(text)
        else:
            return self._process_nltk(text)
    
    def _process_spacy(self, text: str) -> Dict[str, Any]:
        """Process text using SpaCy - FULL NLP capabilities"""
        doc = self.nlp(text)
        
        # 1. TOKENS with POS, lemma, dependency
        tokens = []
        for token in doc:
            tokens.append({
                "text": token.text,
                "lemma": token.lemma_.lower(),
                "pos": token.pos_,          # NOUN, VERB, ADJ, etc.
                "tag": token.tag_,          # Detailed tag (NN, VBD, etc.)
                "dep": token.dep_,          # nsubj, dobj, ROOT, etc.
                "head": token.head.text,    # Head word in dependency
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
                "is_alpha": token.is_alpha
            })
        
        # 2. NAMED ENTITIES (NER)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,        # ORG, PERSON, GPE, PRODUCT, etc.
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # 3. NOUN CHUNKS (compound nouns)
        noun_chunks = []
        for chunk in doc.noun_chunks:
            # Filter out chunks that start with determiners
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                noun_chunks.append({
                    "text": chunk.text,
                    "root": chunk.root.text,
                    "root_pos": chunk.root.pos_
                })
        
        # 4. DEPENDENCIES (for relation extraction)
        dependencies = []
        for token in doc:
            if token.dep_ in ['nsubj', 'nsubjpass', 'dobj', 'pobj', 'attr', 'ROOT']:
                dependencies.append({
                    "child": token.text,
                    "dep": token.dep_,
                    "head": token.head.text,
                    "head_pos": token.head.pos_
                })
        
        # 5. SENTENCES
        sentences = [sent.text for sent in doc.sents]
        
        # 6. LEMMAS (filtered, for topic modeling)
        lemmas = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2
        ]
        
        # 7. CHARACTERS & LOCATIONS (for comic generation)
        characters = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'FAC']]
        
        # 8. NOUNS and PROPER NOUNS (for keyphrase candidates)
        nouns = [
            {"text": token.text, "lemma": token.lemma_, "pos": token.pos_}
            for token in doc 
            if token.pos_ in ['NOUN', 'PROPN'] and not token.is_stop
        ]
        
        # 9. SUBJECTS (for keyphrase importance)
        subjects = [
            token.text for token in doc 
            if token.dep_ in ['nsubj', 'nsubjpass']
        ]
        
        return {
            # Core data
            "original_text": text,
            "cleaned_text": text.strip(),
            "sentences": sentences,
            "word_count": len([t for t in doc if not t.is_punct]),
            "sentence_count": len(sentences),
            
            # NLP Outputs (used by downstream models)
            "tokens": tokens,           # Used by: Classifier, all models
            "entities": entities,       # Used by: Keyphrase Extractor
            "noun_chunks": noun_chunks, # Used by: Keyphrase Extractor
            "dependencies": dependencies, # Used by: Relation Extractor
            "lemmas": lemmas,           # Used by: Topic Modeler
            "nouns": nouns,             # Used by: Keyphrase Extractor
            "subjects": subjects,       # Used by: Keyphrase Extractor
            
            # For comic generation
            "characters": list(set(characters)),
            "locations": list(set(locations)),
            
            # SpaCy doc object (for advanced use)
            "spacy_doc": doc
        }
    
    def _process_nltk(self, text: str) -> Dict[str, Any]:
        """Process text using NLTK - fallback with limited capabilities"""
        # Import NLTK functions
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk import pos_tag, ne_chunk
        from nltk.chunk import tree2conlltags
        
        sentences = sent_tokenize(text)
        
        # Tokenize and POS tag
        tokens = []
        all_words = []
        for sent in sentences:
            words = word_tokenize(sent)
            pos_tags = pos_tag(words)
            all_words.extend(words)
            
            for word, tag in pos_tags:
                tokens.append({
                    "text": word,
                    "lemma": word.lower(),
                    "pos": self._convert_pos_tag(tag),
                    "tag": tag,
                    "dep": "",  # NLTK doesn't have dependency parsing
                    "head": "",
                    "is_stop": word.lower() in self.stop_words,
                    "is_punct": not word.isalnum(),
                    "is_alpha": word.isalpha()
                })
        
        # Extract entities using NLTK NER
        entities = self._extract_entities_nltk(text)
        
        # Extract noun phrases
        noun_chunks = self._extract_noun_phrases_nltk(text)
        
        # Lemmas for topic modeling
        lemmas = [
            t["lemma"] for t in tokens 
            if not t["is_stop"] and not t["is_punct"] and t["is_alpha"] and len(t["text"]) > 2
        ]
        
        # Nouns
        nouns = [
            {"text": t["text"], "lemma": t["lemma"], "pos": t["pos"]}
            for t in tokens 
            if t["pos"] in ['NOUN', 'PROPN'] and not t["is_stop"]
        ]
        
        return {
            "original_text": text,
            "cleaned_text": text.strip(),
            "sentences": sentences,
            "word_count": len([t for t in tokens if not t["is_punct"]]),
            "sentence_count": len(sentences),
            "tokens": tokens,
            "entities": entities,
            "noun_chunks": noun_chunks,
            "dependencies": [],  # NLTK doesn't support this
            "lemmas": lemmas,
            "nouns": nouns,
            "subjects": [],
            "characters": [],
            "locations": [],
            "spacy_doc": None
        }
    
    def _convert_pos_tag(self, tag: str) -> str:
        """Convert Penn Treebank tags to universal tags"""
        tag_map = {
            'NN': 'NOUN', 'NNS': 'NOUN', 'NNP': 'PROPN', 'NNPS': 'PROPN',
            'VB': 'VERB', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB', 'VBP': 'VERB', 'VBZ': 'VERB',
            'JJ': 'ADJ', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RB': 'ADV', 'RBR': 'ADV', 'RBS': 'ADV',
            'PRP': 'PRON', 'PRP$': 'PRON',
        }
        return tag_map.get(tag, 'X')
    
    def _extract_entities_nltk(self, text: str) -> List[Dict]:
        """Extract named entities using NLTK"""
        entities = []
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            named_entities = ne_chunk(pos_tags)
            iob_tags = tree2conlltags(named_entities)
            
            current_entity = []
            current_type = None
            
            for word, pos, tag in iob_tags:
                if tag.startswith('B-'):
                    if current_entity:
                        entities.append({
                            "text": " ".join(current_entity),
                            "label": current_type,
                            "start": 0, "end": 0
                        })
                    current_entity = [word]
                    current_type = tag[2:]
                elif tag.startswith('I-'):
                    current_entity.append(word)
                else:
                    if current_entity:
                        entities.append({
                            "text": " ".join(current_entity),
                            "label": current_type,
                            "start": 0, "end": 0
                        })
                    current_entity = []
            
            if current_entity:
                entities.append({
                    "text": " ".join(current_entity),
                    "label": current_type,
                    "start": 0, "end": 0
                })
        except Exception as e:
            print(f"NER error: {e}")
        
        return entities
    
    def _extract_noun_phrases_nltk(self, text: str) -> List[Dict]:
        """Extract noun phrases using NLTK POS patterns"""
        noun_chunks = []
        try:
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            
            current_np = []
            for word, tag in pos_tags:
                if tag in ['NN', 'NNS', 'NNP', 'NNPS', 'JJ']:
                    current_np.append(word)
                else:
                    if current_np:
                        noun_chunks.append({
                            "text": " ".join(current_np),
                            "root": current_np[-1],
                            "root_pos": "NOUN"
                        })
                        current_np = []
            
            if current_np:
                noun_chunks.append({
                    "text": " ".join(current_np),
                    "root": current_np[-1],
                    "root_pos": "NOUN"
                })
        except Exception as e:
            print(f"Noun phrase error: {e}")
        
        return noun_chunks
