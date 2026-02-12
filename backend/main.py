"""
VisualVerse Backend - FastAPI Server
A Dual-Mode NLP System for Converting Text into Comics and Mind-Maps
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our NLP modules
from nlp.preprocessing.preprocessor import TextPreprocessor
from nlp.classification.classifier import TextClassifier
from nlp.keyphrase.extractor import KeyphraseExtractor
from nlp.topic_model.topic_modeler import TopicModeler
from nlp.relation.relation_extractor import RelationExtractor
from comic_gen.comic_generator import ComicGenerator
from mindmap_gen.mindmap_generator import MindMapGenerator

app = FastAPI(
    title="VisualVerse API",
    description="A Dual-Mode NLP System for Converting Text into Comics and Mind-Maps",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
preprocessor = TextPreprocessor()
classifier = TextClassifier()
keyphrase_extractor = KeyphraseExtractor()
topic_modeler = TopicModeler()
relation_extractor = RelationExtractor()
comic_generator = ComicGenerator()
mindmap_generator = MindMapGenerator()


# Request/Response Models
class TextInput(BaseModel):
    text: str
    mode: Optional[str] = "auto"  # "auto", "comic", "mindmap"


class ProcessingResult(BaseModel):
    mode: str  # "comic" or "mindmap"
    title: str
    summary: str
    comic_data: Optional[List[Dict[str, Any]]] = None
    mindmap_data: Optional[Dict[str, Any]] = None


class TrainingRequest(BaseModel):
    training_model: str  # "keyphrase", "topic", "relation"
    dataset_name: Optional[str] = None
    
    class Config:
        protected_namespaces = ()


class TrainingStatus(BaseModel):
    status: str
    progress: float
    message: str


class ImageRequest(BaseModel):
    prompt: str


# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to VisualVerse API",
        "version": "1.0.0",
        "modes": ["comic", "mindmap"],
        "endpoints": {
            "process": "/api/process",
            "classify": "/api/classify",
            "train": "/api/train/{model_type}",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "preprocessor": preprocessor.is_ready(),
            "classifier": classifier.is_ready(),
            "keyphrase_extractor": keyphrase_extractor.is_ready(),
            "topic_modeler": topic_modeler.is_ready(),
            "relation_extractor": relation_extractor.is_ready()
        }
    }


@app.post("/api/classify")
async def classify_text(input_data: TextInput):
    """Classify text as narrative or informational"""
    try:
        # Preprocess text
        preprocessed = preprocessor.process(input_data.text)
        
        # Classify
        classification = classifier.classify(preprocessed)
        
        return {
            "text_type": classification["type"],
            "confidence": classification["confidence"],
            "features": classification["features"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process", response_model=ProcessingResult)
async def process_text(input_data: TextInput):
    """Main processing endpoint - converts text to comic or mindmap"""
    try:
        # Step 1: Preprocess
        preprocessed = preprocessor.process(input_data.text)
        
        # Step 2: Determine mode
        if input_data.mode == "auto":
            classification = classifier.classify(preprocessed)
            mode = "comic" if classification["type"] == "narrative" else "mindmap"
        else:
            mode = input_data.mode
        
        # Step 3: Generate output based on mode
        if mode == "comic":
            result = await comic_generator.generate(preprocessed)
            return ProcessingResult(
                mode="comic",
                title=result["title"],
                summary=result["summary"],
                comic_data=result["panels"]
            )
        else:
            # Extract more keyphrases for better coverage (increased from 10 to 20)
            keyphrases = keyphrase_extractor.extract(preprocessed, top_k=20)
            
            # Model topics
            topics = topic_modeler.model_topics(preprocessed, keyphrases)
            
            # Add original text to topics for mindmap generation
            topics["original_text"] = preprocessed.get("original_text", "")
            
            # Extract relationships
            relations = relation_extractor.extract(preprocessed, keyphrases)
            
            # Generate mindmap
            mindmap = mindmap_generator.generate(keyphrases, topics, relations)
            
            return ProcessingResult(
                mode="mindmap",
                title=mindmap["title"],
                summary=mindmap["summary"],
                mindmap_data=mindmap["graph"]
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-image")
async def generate_image(request: ImageRequest):
    """Generate a comic panel image using DreamShaper via HF Inference API"""
    try:
        if comic_generator.use_placeholder:
            raise HTTPException(
                status_code=400,
                detail="HF_API_TOKEN not configured. Add it to your .env file."
            )
        
        image_url = comic_generator._call_dreamshaper(request.prompt)
        return {"image_url": image_url}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train/{model_type}")
async def train_model(model_type: str, request: TrainingRequest):
    """Train a specific NLP model"""
    try:
        if model_type == "keyphrase":
            result = await keyphrase_extractor.train(request.dataset_name)
        elif model_type == "topic":
            result = await topic_modeler.train(request.dataset_name)
        elif model_type == "relation":
            result = await relation_extractor.train(request.dataset_name)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        return {
            "status": "success",
            "model_type": model_type,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/status")
async def get_model_status():
    """Get the status of all trained models"""
    return {
        "keyphrase_model": {
            "trained": keyphrase_extractor.is_trained(),
            "metrics": keyphrase_extractor.get_metrics()
        },
        "topic_model": {
            "trained": topic_modeler.is_trained(),
            "metrics": topic_modeler.get_metrics()
        },
        "relation_model": {
            "trained": relation_extractor.is_trained(),
            "metrics": relation_extractor.get_metrics()
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print(" VisualVerse Backend Server")
    print(" API Docs: http://localhost:8000/docs")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
