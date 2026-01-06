# VisualVerse: Dual-Mode NLP System
## Complete Project Report

---

# 1. Abstract

**VisualVerse** is an intelligent text-visualization system that transforms text into two visual formats:
- **Comic Strips** → For narrative/story-based text
- **Mind-Maps** → For informational/conceptual text

The system uses **Natural Language Processing (NLP)** for text understanding, classification, and extraction, combined with **visualization techniques** for output generation.

---

# 2. Problem Statement

| Problem | Impact |
|---------|--------|
| Long paragraphs are hard to process | Low comprehension |
| Abstract concepts are difficult to visualize | Poor retention |
| No tool converts text to both comics AND mind-maps | Gap in learning tools |

**Solution**: Build a dual-mode system that automatically detects text type and generates appropriate visual output.

---

# 3. Objectives

### Primary
- ✅ Build NLP-driven text → visual transformation system
- ✅ Automate story segmentation & comic panel creation
- ✅ Automate keyphrase extraction & mind-map generation
- ✅ User-friendly web interface

### Secondary
- ✅ Support visual/multimodal learning
- ✅ Help students simplify complex content
- ✅ Foundation for academic research

---

# 4. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                               │
│                    (Story or Concept Text)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NLP PREPROCESSING                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │Tokenize  │→│POS Tag   │→│NER       │→│Dependency Parse  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TEXT CLASSIFICATION                            │
│            (LSTM + Feature-based Classifier)                     │
│                               │                                  │
│              ┌────────────────┼────────────────┐                │
│              ▼                                 ▼                │
│      ┌──────────────┐                 ┌──────────────┐          │
│      │  NARRATIVE   │                 │INFORMATIONAL │          │
│      │   (Story)    │                 │  (Concept)   │          │
│      └──────────────┘                 └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
              │                                 │
              ▼                                 ▼
┌──────────────────────┐           ┌──────────────────────┐
│   COMIC GENERATOR    │           │  MINDMAP GENERATOR   │
│                      │           │                      │
│ • Scene extraction   │           │ • Keyphrase extract  │
│ • Character detect   │           │ • Topic modeling     │
│ • Panel creation     │           │ • Relation mapping   │
│ • Layout generation  │           │ • Graph construction │
└──────────────────────┘           └──────────────────────┘
              │                                 │
              ▼                                 ▼
┌──────────────────────┐           ┌──────────────────────┐
│    COMIC STRIP       │           │      MIND-MAP        │
│   (Visual Panels)    │           │  (Interactive Graph) │
└──────────────────────┘           └──────────────────────┘
```

---

# 5. NLP Concepts Used (Unit-wise)

## Unit 1: Computational Linguistics

| Concept | Where Used | Implementation |
|---------|-----------|----------------|
| **Tokenization** | Preprocessing | SpaCy tokenizer splits text into tokens |
| **Sentence Splitting** | Scene detection | SpaCy's sentence boundary detection |
| **Morphology** | Feature extraction | Lemmatization for keyphrase normalization |
| **Syntax** | Dependency parsing | Subject-verb-object extraction for relations |
| **Semantics** | Topic modeling | Word embeddings for semantic similarity |

## Unit 2: Word Representation

| Technique | Where Used | Purpose |
|-----------|-----------|---------|
| **TF-IDF** | Keyphrase extraction | Score important terms |
| **Word Embeddings** | Topic clustering | Semantic similarity grouping |
| **Bag of Words** | Text classification | Feature vector creation |

## Unit 3: Deep Learning for NLP

| Model | Where Used | Purpose |
|-------|-----------|---------|
| **LSTM** | Text classifier | Narrative vs Informational detection |
| **BERT/Transformers** | KeyBERT | Keyphrase extraction with embeddings |
| **Attention** | Sentence importance | Weight important sentences |

## Unit 4: NLP Applications

| Application | Implementation |
|-------------|----------------|
| **NER** | SpaCy - Extract characters, locations, objects |
| **POS Tagging** | SpaCy - Identify nouns, verbs, adjectives |
| **Dependency Parsing** | SpaCy - Extract subject-verb-object relations |
| **Keyphrase Extraction** | KeyBERT + YAKE + NLP-based hybrid |
| **Topic Modeling** | LDA + Semantic clustering |

---

# 6. NLP Pipeline Explained

## 6.1 Preprocessing Pipeline

```python
# File: backend/nlp/preprocessing/preprocessor.py

def preprocess(text):
    # Step 1: Load SpaCy model
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    
    # Step 2: Tokenization
    tokens = [token.text for token in doc]
    
    # Step 3: POS Tagging
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    # Step 4: Named Entity Recognition
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # Labels: PERSON, ORG, GPE (location), DATE, etc.
    
    # Step 5: Dependency Parsing
    dependencies = [(token.text, token.dep_, token.head.text) 
                    for token in doc]
    
    return {
        "tokens": tokens,
        "pos_tags": pos_tags,
        "entities": entities,
        "dependencies": dependencies
    }
```

## 6.2 Text Classification

```python
# File: backend/nlp/classification/lstm_classifier.py

# Features used for classification:
# 1. Narrative indicators: pronouns (he, she, they), past tense verbs
# 2. Story markers: "once upon", "said", "walked", action verbs
# 3. Dialogue patterns: quotation marks, speech verbs
# 4. Named entities: PERSON count (stories have characters)

# Classification logic:
if narrative_score > 0.5:
    return "comic"  # → Comic Generator
else:
    return "mindmap"  # → Mind-Map Generator
```

## 6.3 Keyphrase Extraction Pipeline

```python
# File: backend/nlp/keyphrase/keyphrase_extractor.py

def extract_keyphrases(text):
    keyphrases = []
    
    # Method 1: Named Entities (PERSON, ORG, GPE, PRODUCT)
    for ent in doc.ents:
        keyphrases.append(ent.text)
    
    # Method 2: Noun Chunks (compound nouns)
    for chunk in doc.noun_chunks:
        keyphrases.append(chunk.text)
    
    # Method 3: Dependency-based (subjects, objects)
    for token in doc:
        if token.dep_ in ["nsubj", "dobj", "pobj"]:
            keyphrases.append(token.text)
    
    # Method 4: POS-based (proper nouns, technical terms)
    for token in doc:
        if token.pos_ in ["PROPN", "NOUN"] and token.is_alpha:
            keyphrases.append(token.lemma_)
    
    # Score and rank keyphrases
    return rank_by_frequency_and_position(keyphrases)
```

## 6.4 Topic Modeling

```python
# File: backend/nlp/topic_model/topic_modeler.py

def extract_topics(text):
    # Method 1: LDA (Latent Dirichlet Allocation)
    # - Treats document as mixture of topics
    # - Each topic is distribution over words
    
    # Method 2: Semantic Clustering
    # - Group related keyphrases by word similarity
    # - Categories like: Frontend, Backend, Database, etc.
    
    return {
        "main_topic": "Web Development",
        "categories": ["Frontend", "Backend", "Database"],
        "topic_distribution": {...}
    }
```

## 6.5 Relation Extraction

```python
# File: backend/nlp/relation/relation_extractor.py

def extract_relations(text):
    relations = []
    
    for token in doc:
        # Subject-Verb-Object pattern
        if token.dep_ == "ROOT":  # Main verb
            subject = [t for t in token.lefts if t.dep_ == "nsubj"]
            object = [t for t in token.rights if t.dep_ == "dobj"]
            
            if subject and object:
                relations.append({
                    "source": subject[0].text,
                    "relation": token.text,
                    "target": object[0].text
                })
    
    return relations
```

---

# 7. Mind-Map Generator

```python
# File: backend/mindmap_gen/mindmap_generator.py

class MindMapGenerator:
    def generate(self, keyphrases, topics, relations):
        # Step 1: Create graph
        graph = networkx.DiGraph()
        
        # Step 2: Add center node (main topic)
        graph.add_node("center", label=main_topic, level=0)
        
        # Step 3: Add category nodes (level 1)
        for category in categories:
            graph.add_node(cat_id, label=category, level=1)
            graph.add_edge("center", cat_id)
        
        # Step 4: Add detail nodes (level 2)
        for keyphrase in keyphrases:
            graph.add_node(det_id, label=keyphrase, level=2)
            graph.add_edge(parent_category, det_id)
        
        # Step 5: Calculate layout positions
        positions = calculate_hierarchical_layout()
        
        return {"nodes": nodes, "edges": edges}
```

### Layout Algorithm:
- **Level 0**: Main topic at top center
- **Level 1**: Categories spread horizontally
- **Level 2**: Details evenly distributed at bottom

---

# 8. Comic Generator

```python
# File: backend/comic_gen/comic_generator.py

class ComicGenerator:
    def generate(self, text, scenes):
        panels = []
        
        for scene in scenes:
            # Extract scene elements
            characters = scene["entities"]["PERSON"]
            location = scene["entities"]["GPE"]
            action = scene["main_verb"]
            
            # Generate panel prompt
            prompt = f"{characters} {action} in {location}"
            
            # Create panel
            panels.append({
                "prompt": prompt,
                "caption": scene["sentence"]
            })
        
        return panels
```

---

# 9. Training Details

## What We Trained

| Component | Training Data | Purpose |
|-----------|--------------|---------|
| Text Classifier | Stories + Wikipedia | Distinguish narrative vs informational |
| Keyphrase Extractor | Academic abstracts | Extract important terms |
| Topic Model | BBC News, WikiHow | Cluster related concepts |

## Why Training Was Needed

- **Classifier**: Pre-trained models don't differentiate story vs concept text
- **Keyphrase**: Domain-specific extraction needed
- **Topics**: Custom category grouping for mind-maps

## Training Files

```
training/
├── keyphrase_training/
│   └── train_keyphrase.py      # Train keyphrase extractor
├── relation_training/
│   └── train_relation.py       # Train relation extractor
├── topic_training/
│   └── train_topic.py          # Train topic model
└── data/
    ├── download_datasets.py    # Download training data
    └── prepare_datasets.py     # Preprocess datasets
```

---

# 10. Tech Stack

## Backend
| Technology | Purpose |
|------------|---------|
| **Python 3.11** | Core language |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |

## NLP Libraries
| Library | Purpose |
|---------|---------|
| **SpaCy** | Tokenization, NER, POS, Dependencies |
| **NLTK** | Additional NLP utilities |
| **Transformers** | BERT models for embeddings |
| **KeyBERT** | Keyphrase extraction |
| **Gensim** | Word2Vec, topic modeling |
| **scikit-learn** | TF-IDF, LDA, clustering |

## Visualization
| Library | Purpose |
|---------|---------|
| **NetworkX** | Graph data structure |
| **PyVis** | Interactive visualization |

## Frontend
| Technology | Purpose |
|------------|---------|
| **React** | UI framework |
| **TypeScript** | Type-safe JavaScript |
| **Vite** | Build tool |
| **Lucide React** | Icons |

## Deployment
| Service | Purpose |
|---------|---------|
| **Render** | Cloud hosting |
| **Docker** | Containerization |

---

# 11. Project Structure

```
VisualVerse/
├── backend/
│   ├── main.py                 # FastAPI app entry
│   ├── api/
│   │   └── routes.py           # API endpoints
│   ├── nlp/
│   │   ├── preprocessing/      # Tokenization, NER, POS
│   │   ├── classification/     # Text classifier (LSTM)
│   │   ├── keyphrase/          # Keyphrase extraction
│   │   ├── topic_model/        # LDA, clustering
│   │   └── relation/           # Relation extraction
│   ├── mindmap_gen/
│   │   └── mindmap_generator.py
│   └── comic_gen/
│       └── comic_generator.py
│
├── training/                   # Model training scripts
├── services/
│   └── geminiService.ts        # API client
├── components/
│   └── Button.tsx              # UI components
├── App.tsx                     # Main React app
└── render.yaml                 # Deployment config
```

---

# 12. API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/api/process` | POST | Main processing endpoint |

### Request Format:
```json
{
    "text": "Your input text here",
    "mode": "auto" | "comic" | "mindmap"
}
```

### Response Format:
```json
{
    "mode": "mindmap",
    "title": "Main Topic",
    "summary": "Description",
    "mindmap_data": {
        "nodes": [...],
        "edges": [...]
    }
}
```

---

# 13. Key Innovations

| Innovation | Description |
|------------|-------------|
| **Dual-Mode Output** | Single system produces both comics and mind-maps |
| **Automatic Classification** | No manual selection needed |
| **Dynamic Mind-Maps** | Node count varies with content complexity |
| **NLP-Driven** | Uses multiple NLP techniques, not just API calls |
| **Hierarchical Layout** | 3-level structure for clear visualization |

---

# 14. Evaluation Metrics

## For Mind-Maps
- **Keyphrase Accuracy**: Are extracted terms relevant?
- **Clustering Quality**: Are related concepts grouped?
- **Graph Connectivity**: Proper parent-child relationships?
- **Layout Clarity**: No overlapping nodes?

## For Comics
- **Story Alignment**: Panels match story flow?
- **Scene Relevance**: Correct characters/locations extracted?
- **Panel Consistency**: Visual coherence across panels?

---

# 15. Datasets Used

| Dataset | Purpose | Size |
|---------|---------|------|
| **ROCStories** | Story understanding | 100K+ stories |
| **WikiHow** | Instructional text | Millions of articles |
| **BBC News** | Topic classification | Thousands of articles |
| **COCO Captions** | Image-text alignment | 5 captions per image |

---

# 16. Future Enhancements

- [ ] Multilingual support
- [ ] Voice input → visual output
- [ ] Custom comic styles (anime, realistic)
- [ ] Character memory for long stories
- [ ] PDF/PNG export
- [ ] Collaborative editing

---

# 17. Conclusion

**VisualVerse** successfully demonstrates:

1. ✅ **NLP Integration**: Uses tokenization, NER, POS, dependency parsing, topic modeling
2. ✅ **Dual-Mode System**: Automatically routes to comic or mind-map
3. ✅ **Clean Architecture**: Modular, maintainable code structure
4. ✅ **Modern Tech Stack**: FastAPI + React + SpaCy + NetworkX
5. ✅ **Cloud Deployment**: Production-ready on Render



---

## Authors
**Ghanasree S** - VisualVerse Project

---


