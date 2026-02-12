"""
Comic Generator Module
Generates comic strips from narrative text using story segmentation and image generation
"""

import os
import base64
import logging
from typing import Dict, Any, List
import requests
from io import BytesIO

logger = logging.getLogger(__name__)

# Text-to-image models on Hugging Face (in priority order)
HF_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "black-forest-labs/FLUX.1-schnell",
]
HF_ROUTER_URL = "https://router.huggingface.co/hf-inference/models"


class ComicGenerator:
    """
    Comic Strip Generator using Stable Diffusion XL via Hugging Face Inference API
    
    Pipeline:
    1. Segment story into scenes/beats
    2. Extract characters and settings for each scene
    3. Generate DreamShaper-style optimized prompts
    4. Generate images via HF Inference API (SDXL / FLUX.1)
    5. Combine panels into comic strip layout
    """
    
    def __init__(self):
        """Initialize the comic generator"""
        self.hf_token = os.getenv("HF_API_TOKEN", "")
        self.use_placeholder = not bool(self.hf_token)
        
    async def generate(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comic strip from preprocessed text
        
        Returns:
            Dict with:
            - title: Story title
            - summary: Brief summary
            - panels: List of panel dicts with caption, image_url, characters, setting
        """
        text = preprocessed.get("original_text", "")
        sentences = preprocessed.get("sentences", [])
        characters = preprocessed.get("characters", [])
        locations = preprocessed.get("locations", [])
        
        # Generate title
        title = self._generate_title(text, preprocessed)
        
        # Generate summary
        summary = self._generate_summary(sentences)
        
        # Segment into panels (4-6 panels typically)
        panels = self._segment_into_panels(sentences, characters, locations)
        
        # Generate images for each panel
        for panel in panels:
            panel["image_url"] = await self._generate_panel_image(panel)
        
        return {
            "title": title,
            "summary": summary,
            "panels": panels
        }
    
    def _generate_title(self, text: str, preprocessed: Dict[str, Any]) -> str:
        """Generate a title for the comic"""
        characters = preprocessed.get("characters", [])
        
        # Use first character's name if available
        if characters:
            return f"The Story of {characters[0]}"
        
        # Extract first significant noun phrase
        noun_phrases = preprocessed.get("noun_phrases", [])
        if noun_phrases:
            return f"A Tale of {noun_phrases[0].title()}"
        
        # Default title
        first_words = text.split()[:5]
        return " ".join(first_words) + "..."
    
    def _generate_summary(self, sentences: List[str]) -> str:
        """Generate a brief summary"""
        if not sentences:
            return "A visual story unfolds..."
        
        # Use first sentence as summary
        summary = sentences[0]
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
    
    def _segment_into_panels(self, sentences: List[str], 
                            characters: List[str], 
                            locations: List[str]) -> List[Dict[str, Any]]:
        """
        Segment story into comic panels.
        
        Panel count scales with input size:
        - 1-2 sentences  → 1-2 panels (one per sentence)
        - 3-5 sentences  → 3 panels
        - 6-9 sentences  → 4 panels
        - 10-15 sentences → 5-6 panels
        - 16-24 sentences → 6-8 panels
        - 25+ sentences  → 8-10 panels
        """
        n = len(sentences)
        
        if n <= 2:
            num_panels = n
        elif n <= 5:
            num_panels = 3
        elif n <= 9:
            num_panels = 4
        elif n <= 15:
            num_panels = min(6, max(5, n // 3))
        elif n <= 24:
            num_panels = min(8, max(6, n // 3))
        else:
            num_panels = min(10, max(8, n // 4))
        
        if len(sentences) < num_panels:
            # If too few sentences, use each sentence as a panel
            panels = []
            for i, sent in enumerate(sentences):
                panels.append(self._create_panel(i + 1, sent, characters, locations))
            return panels
        
        # Distribute sentences across panels
        sentences_per_panel = len(sentences) // num_panels
        panels = []
        
        for i in range(num_panels):
            start_idx = i * sentences_per_panel
            end_idx = start_idx + sentences_per_panel if i < num_panels - 1 else len(sentences)
            
            panel_text = " ".join(sentences[start_idx:end_idx])
            panels.append(self._create_panel(i + 1, panel_text, characters, locations))
        
        return panels
    
    def _create_panel(self, panel_num: int, text: str, 
                     characters: List[str], locations: List[str]) -> Dict[str, Any]:
        """Create a single panel structure"""
        # Generate image prompt from text
        prompt = self._generate_image_prompt(text, characters, locations)
        
        # Use full text as caption (no truncation)
        caption = text
        
        return {
            "id": f"panel_{panel_num}",
            "panel_number": panel_num,
            "caption": caption,
            "full_text": text,
            "prompt": prompt,
            "characters": characters,
            "setting": locations[0] if locations else "Unknown location",
            "image_url": None  # Will be filled by image generation
        }
    
    def _generate_image_prompt(self, text: str, 
                               characters: List[str], 
                               locations: List[str]) -> str:
        """
        Generate a DreamShaper-optimized prompt for the panel.
        
        DreamShaper excels with detailed, descriptive prompts including
        style keywords, quality boosters, and negative prompt hints.
        """
        # DreamShaper quality boosters
        quality = "masterpiece, best quality, highly detailed"
        style = "comic book art style, vibrant colors, dynamic composition, cel shading, bold outlines"
        
        # Scene description (simplified from text)
        scene = text[:180] if len(text) > 180 else text
        
        # Characters
        char_desc = ""
        if characters:
            char_desc = f"featuring {', '.join(characters[:2])}, "
        
        # Location
        loc_desc = ""
        if locations:
            loc_desc = f"set in {locations[0]}, "
        
        # Mood/atmosphere
        mood = self._detect_mood(text)
        mood_desc = f"{mood} atmosphere, cinematic lighting"
        
        prompt = f"{quality}, {style}, {char_desc}{loc_desc}{scene}, {mood_desc}"
        
        return prompt
    
    def _detect_mood(self, text: str) -> str:
        """Detect the mood/atmosphere of the text"""
        text_lower = text.lower()
        
        # Check for mood indicators
        if any(word in text_lower for word in ["happy", "joy", "laugh", "smile", "excited"]):
            return "cheerful and bright"
        elif any(word in text_lower for word in ["sad", "cry", "tears", "lonely", "grief"]):
            return "melancholic and somber"
        elif any(word in text_lower for word in ["angry", "rage", "fight", "battle", "war"]):
            return "intense and dramatic"
        elif any(word in text_lower for word in ["fear", "dark", "scary", "horror", "terror"]):
            return "dark and mysterious"
        elif any(word in text_lower for word in ["love", "romance", "heart", "kiss"]):
            return "romantic and warm"
        elif any(word in text_lower for word in ["adventure", "journey", "discover", "explore"]):
            return "adventurous and exciting"
        else:
            return "neutral and balanced"
    
    async def _generate_panel_image(self, panel: Dict[str, Any]) -> str:
        """
        Generate image for a panel using DreamShaper via HF Inference API.
        Falls back to SVG placeholder if API is unavailable.
        """
        if self.use_placeholder:
            return self._get_placeholder_image(panel["panel_number"], panel.get("caption", ""))
        
        try:
            return self._call_dreamshaper(panel["prompt"])
        except Exception as e:
            logger.warning(f"DreamShaper API failed for panel {panel['panel_number']}: {e}")
            return self._get_placeholder_image(panel["panel_number"], panel.get("caption", ""))
    
    def _call_dreamshaper(self, prompt: str) -> str:
        """
        Call Hugging Face Inference API with SDXL / FLUX.1 models.
        Tries models in priority order. Returns base64-encoded PNG data URI.
        """
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "negative_prompt": "blurry, bad anatomy, bad hands, cropped, worst quality, low quality, watermark, text, signature, deformed",
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
            }
        }
        
        last_error = None
        for model in HF_MODELS:
            url = f"{HF_ROUTER_URL}/{model}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=120)
                
                if response.status_code == 200:
                    img_bytes = response.content
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    logger.info(f"Image generated successfully with {model}")
                    return f"data:image/png;base64,{b64}"
                else:
                    last_error = f"{model} returned {response.status_code}: {response.text[:100]}"
                    logger.warning(last_error)
            except requests.exceptions.Timeout:
                last_error = f"{model} timed out"
                logger.warning(last_error)
            except Exception as e:
                last_error = f"{model} error: {e}"
                logger.warning(last_error)
        
        raise RuntimeError(f"All HF models failed. Last error: {last_error}")
    
    def _get_placeholder_image(self, panel_number: int, caption: str = "") -> str:
        """
        Generate a beautiful SVG placeholder image
        Fast and works without external API
        """
        colors = [
            ("#FF6B6B", "#C44D4D"),  # Red
            ("#4ECDC4", "#36A89F"),  # Teal
            ("#45B7D1", "#2E8DA8"),  # Blue
            ("#96CEB4", "#6BAF8F"),  # Green
            ("#FFEAA7", "#D4C680"),  # Yellow
            ("#DDA0DD", "#B87AB8"),  # Purple
        ]
        
        color, dark_color = colors[(panel_number - 1) % len(colors)]
        
        # Shorten caption for display
        short_caption = caption[:50] + "..." if len(caption) > 50 else caption
        
        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512" viewBox="0 0 512 512">
            <defs>
                <linearGradient id="bg{panel_number}" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:{color}"/>
                    <stop offset="100%" style="stop-color:{dark_color}"/>
                </linearGradient>
            </defs>
            <rect width="512" height="512" fill="url(#bg{panel_number})"/>
            <rect x="10" y="10" width="492" height="492" fill="none" stroke="white" stroke-width="4" rx="10"/>
            <text x="256" y="180" font-family="Comic Sans MS, cursive, sans-serif" font-size="72" fill="white" text-anchor="middle" opacity="0.9">🎬</text>
            <text x="256" y="280" font-family="Arial, sans-serif" font-size="36" fill="white" text-anchor="middle" font-weight="bold">PANEL {panel_number}</text>
            <text x="256" y="380" font-family="Arial, sans-serif" font-size="16" fill="white" text-anchor="middle" opacity="0.8">{short_caption}</text>
            <text x="256" y="470" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle" opacity="0.5">VisualVerse Comic Generator</text>
        </svg>'''
        
        # Return as base64 encoded SVG
        import base64
        svg_bytes = svg.encode('utf-8')
        b64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f"data:image/svg+xml;base64,{b64}"
    
    def generate_comic_layout(self, panels: List[Dict[str, Any]], 
                             layout: str = "grid") -> Dict[str, Any]:
        """
        Generate final comic layout configuration
        
        Layouts:
        - grid: 2x2 or 2x3 grid
        - vertical: Single column
        - manga: Right-to-left reading
        """
        num_panels = len(panels)
        
        if layout == "grid":
            if num_panels <= 2:
                rows, cols = 1, num_panels
            elif num_panels <= 4:
                rows, cols = 2, 2
            elif num_panels <= 6:
                rows, cols = 2, 3
            elif num_panels <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 4, 3
        elif layout == "vertical":
            rows, cols = num_panels, 1
        elif layout == "manga":
            rows, cols = 2, 2  # Will be reversed in frontend
        else:
            rows, cols = 2, 2
        
        return {
            "layout": layout,
            "rows": rows,
            "cols": cols,
            "panel_order": list(range(len(panels))),
            "reading_direction": "rtl" if layout == "manga" else "ltr"
        }
