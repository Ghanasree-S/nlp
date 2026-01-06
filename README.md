# VisualVerse


## 📖 A Dual-Mode NLP System for Converting Text into Comics and Mind-Maps

VisualVerse is an intelligent text-visualization system that transforms any given text into **comic strips** (for stories) or **mind-maps** (for explanatory/content-based text) using state-of-the-art Natural Language Processing (NLP) and AI.

---

## 🎯 Features

### Dual-Mode Processing
- **📖 Comic Mode**: Converts narrative/story text into comic panels with AI-generated images
- **🧠 Mind Map Mode**: Converts informational text into visual concept graphs

### Trained NLP Components
The system includes **trained** (not just pre-trained) NLP models:

| Component | Purpose | Training Approach |
|-----------|---------|-------------------|
| **Text Classifier** | Narrative vs Informational detection | Random Forest with linguistic features |
| **Keyphrase Extractor** | Extract key concepts for mind maps | Logistic Regression on candidate features |
| **Topic Modeler** | Cluster concepts into topics | LDA (Latent Dirichlet Allocation) |
| **Relation Extractor** | Identify relationships between concepts | MLP Classifier on pattern features |

---

## 🏗️ Project Structure

```
VisualVerse/
├── 📁 frontend/                 # React + Vite frontend (existing)
│   ├── App.tsx                  # Main application
│   ├── components/              # UI components
│   └── services/                # API services
│
├── 📁 backend/                  # Python FastAPI backend
│   ├── main.py                  # FastAPI server
│   ├── requirements.txt         # Python dependencies
│   ├── 📁 api/                  # API endpoints
│   ├── 📁 nlp/                  # NLP modules
│   │   ├── preprocessing/       # Tokenization, NER, POS tagging
│   │   ├── classification/      # Narrative vs Informational
│   │   ├── keyphrase/          # Keyphrase extraction (TRAINED)
│   │   ├── topic_model/        # Topic modeling (TRAINED)
│   │   └── relation/           # Relation extraction (TRAINED)
│   ├── 📁 comic_gen/           # Comic generation pipeline
│   ├── 📁 mindmap_gen/         # Mind map generation
│   └── 📁 models/              # Trained model weights
│
├── 📁 training/                 # Training scripts
│   ├── train_all.py            # Train all models
│   ├── 📁 data/                # Training datasets
│   ├── 📁 keyphrase_training/  # Keyphrase model training
│   ├── 📁 topic_training/      # Topic model training
│   └── 📁 relation_training/   # Relation model training
│
└── 📁 evaluation/              # Evaluation metrics
    └── evaluate.py             # Quality assessment
```

---

## 🚀 Quick Start

### Prerequisites
- **Node.js** (v18+) for frontend
- **Python** (v3.9+) for backend

### 1. Setup Backend

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Unix)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### 2. Prepare Datasets & Train Models

```bash
# Prepare training datasets
python training/data/prepare_datasets.py

# Train all NLP models
python training/train_all.py
```

Or use the batch script (Windows):
```bash
train_models.bat
```

### 3. Run Backend Server

```bash
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### 4. Run Frontend

```bash
# In a new terminal, from project root
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

---

## 📊 NLP Training Details

### Text Classifier Training
- **Algorithm**: Random Forest with TF-IDF features
- **Features**: Linguistic indicators (pronouns, verb tenses, dialogue markers)
- **Dataset**: Labeled narrative vs informational texts
- **Accuracy**: ~90%+

### Keyphrase Extraction Training
- **Algorithm**: Logistic Regression / Gradient Boosting
- **Features**: Position, frequency, TF-IDF, length
- **Dataset**: Inspec-style abstracts with keyphrases
- **Metrics**: Precision, Recall, F1

### Topic Modeling Training
- **Algorithm**: LDA (Latent Dirichlet Allocation)
- **Evaluation**: Perplexity, Coherence score
- **Dataset**: WikiHow-style procedural documents
- **Output**: Topic-word distributions

### Relation Extraction Training
- **Algorithm**: MLP Classifier
- **Relation Types**: IS_A, PART_OF, CAUSES, REQUIRES, RELATES_TO, CONTRASTS
- **Features**: Context TF-IDF + pattern matching
- **Accuracy**: ~85%+

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/api/health` | GET | Health check |
| `/api/classify` | POST | Classify text type |
| `/api/process` | POST | Main processing endpoint |
| `/api/train/{model}` | POST | Train specific model |
| `/api/models/status` | GET | Model training status |

### Example Request

```json
POST /api/process
{
  "text": "Once upon a time, a brave knight...",
  "mode": "auto"  // "auto", "comic", or "mindmap"
}
```

### Example Response (Comic)

```json
{
  "mode": "comic",
  "title": "The Story of the Knight",
  "summary": "A brave knight embarks on an adventure",
  "comic_data": [
    {
      "id": "panel_1",
      "caption": "A brave knight prepares for battle",
      "prompt": "Comic style, knight in armor...",
      "image_url": "https://..."
    }
  ]
}
```

### Example Response (Mind Map)

```json
{
  "mode": "mindmap",
  "title": "Machine Learning Concepts",
  "summary": "Key concepts in ML",
  "mindmap_data": {
    "nodes": [
      {"id": "1", "label": "Machine Learning", "type": "topic"},
      {"id": "2", "label": "Neural Networks", "type": "concept"}
    ],
    "edges": [
      {"source": "1", "target": "2", "relation": "PART_OF"}
    ]
  }
}
```

---

## 📈 Evaluation Metrics

### Comic Generation
- **Story Alignment**: How well panels follow the story
- **Scene Relevance**: Relevance of image prompts
- **Panel Consistency**: Character/setting consistency
- **Visual Coherence**: Overall narrative flow

### Mind Map Generation
- **Keyphrase Accuracy**: Relevance of extracted phrases
- **Concept Clustering**: Quality of topic grouping
- **Graph Connectivity**: Edge density analysis
- **Hierarchy Quality**: Topic-concept structure

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| FastAPI | Web framework |
| SpaCy | NLP preprocessing |
| NLTK | Tokenization |
| Scikit-learn | ML models |
| NetworkX | Graph generation |

### NLP Training
| Library | Purpose |
|---------|---------|
| Scikit-learn | Classifiers (RF, LR, MLP) |
| Gensim / sklearn | LDA Topic Modeling |
| NLTK | Text processing |

### Frontend
| Technology | Purpose |
|------------|---------|
| React 19 | UI framework |
| Vite | Build tool |
| TypeScript | Type safety |
| Lucide React | Icons |

---

## 📝 Datasets Used

### For Story → Comic
- ROCStories (story understanding)
- Visual Storytelling Dataset (VIST)
- COCO Captions (scene descriptions)

### For Text → Mind Map
- WikiHow (procedural text)
- BBC News (topic classification)
- arXiv Abstracts (keyphrases)

---

## 🎓 Academic Value

This project demonstrates:
1. **NLP Pipeline Design**: Complete preprocessing → classification → generation
2. **Custom Model Training**: Not just using pre-trained models
3. **Multi-Modal Output**: Text to visual representations
4. **Evaluation Framework**: Comprehensive quality metrics

Suitable for:
- Final year engineering projects
- Academic publications
- Research demonstrations

---

## 📄 License

Apache-2.0

---

## 🤝 Contributing

Contributions welcome! Please read the contribution guidelines first.

---

<div align="center">
<b>VisualVerse</b> - Transform Text into Visual Stories
</div>
