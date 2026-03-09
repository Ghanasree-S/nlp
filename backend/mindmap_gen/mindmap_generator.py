"""
MindMap Generator - Intelligent Hierarchical Layout
Creates educational mindmaps with proper concept clustering:
- Level 0: Main Topic (center)
- Level 1: Categories (grouped by semantic similarity & topic modeling)
- Level 2: Sub-concepts (organized under relevant categories)
- Uses NLP-extracted keyphrases + relation data for smart hierarchy

Features:
- Semantic clustering of related concepts
- Relation-based edge labeling
- Proper radial layout with readable spacing
"""

import networkx as nx
from typing import Dict, Any, List, Set
import math
from collections import defaultdict


class MindMapGenerator:
    """Intelligent Mind Map Generator with semantic hierarchy"""
    
    def __init__(self):
        self.graph = None
        self.relation_map = {}  # Map of concept pairs to relations
        
    def generate(self, keyphrases: List[Dict[str, Any]], 
                 topics: Dict[str, Any], 
                 relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate intelligent hierarchical mind map using NLP insights"""
        self.graph = nx.DiGraph()
        
        # Extract main topic intelligently
        original_text = topics.get("original_text", "")
        main_topic = self._extract_main_topic(original_text)
        
        # Build relation map for smarter hierarchy
        self._build_relation_map(relations, keyphrases)
        
        # Smart organization: use topics + relations to group concepts
        categories, details_map = self._organize_with_semantics(
            keyphrases, main_topic, topics
        )
        
        # Build hierarchical structure with proper connections
        self._build_hierarchy(main_topic, categories, details_map, relations)
        
        # Calculate positions for readability
        layout = self._calculate_layout()
        
        # Build output for frontend
        output = self._build_output(layout)
        
        return {
            "title": main_topic,
            "summary": f"Mind map: {main_topic}",
            "graph": output,
            "stats": {
                "node_count": self.graph.number_of_nodes(),
                "edge_count": self.graph.number_of_edges(),
                "categories": len(categories)
            }
        }
    
    def _build_relation_map(self, relations: List[Dict[str, Any]], 
                           keyphrases: List[Dict[str, Any]]) -> None:
        """Build a map of concept pairs to their relationship types"""
        phrase_set = {kp.get("phrase", "").lower() for kp in keyphrases if kp.get("phrase")}
        
        for rel in relations:
            src = rel.get("source", "").lower()
            tgt = rel.get("target", "").lower()
            rel_type = rel.get("type", "RELATES_TO")
            
            if src in phrase_set and tgt in phrase_set:
                key = (src, tgt)
                self.relation_map[key] = rel_type
    
    def _organize_with_semantics(self, keyphrases: List[Dict[str, Any]], 
                                 main_topic: str, 
                                 topics: Dict[str, Any]) -> tuple:
        """
        Organize keyphrases using semantic clustering and topic modeling
        - Uses topic distribution to group related concepts
        - Categories are the main topics extracted
        - Details are keyphrase organized by relevance to categories
        """
        if not keyphrases:
            return self._get_default_categories(topics.get("original_text", "")), {}
        
        # Get topic distribution and original text
        topic_dists = topics.get("topic_distribution", {})
        original_text = topics.get("original_text", "")
        
        # Merge compound phrases (e.g. "carbon" + "dioxide" → "carbon dioxide")
        keyphrases = self._merge_compound_keyphrases(keyphrases, original_text)
        
        # Clean and score keyphrases
        cleaned_kps = self._clean_keyphrases(keyphrases, main_topic)
        
        if not cleaned_kps:
            return self._get_default_categories(original_text), {}
        
        # Smart category extraction from context
        categories = self._smart_extract_categories(cleaned_kps, original_text, topic_dists)
        
        # Smart mapping of details to categories
        details_map = self._smart_map_details(cleaned_kps, categories, original_text)
        
        return categories, details_map
    
    # Single-word terms that are too generic to be useful detail nodes
    GENERIC_SINGLE_WORDS = {
        'process', 'energy', 'system', 'systems', 'method', 'methods',
        'type', 'types', 'form', 'forms', 'way', 'ways', 'part', 'parts',
        'role', 'result', 'results', 'use', 'case', 'cases',
        'level', 'levels', 'state', 'states', 'model', 'models',
        'step', 'steps', 'stage', 'stages', 'point', 'points',
        'rate', 'rates', 'factor', 'factors', 'effect', 'effects',
        'change', 'changes', 'group', 'groups', 'area', 'areas',
        'number', 'amount', 'value', 'time', 'data', 'information',
        'reactions', 'reaction', 'application', 'applications',
    }

    # Known compound terms to merge when found as split single-word keyphrases
    COMPOUND_TERMS = [
        ('carbon', 'dioxide'), ('carbon', 'cycle'), ('calvin', 'cycle'),
        ('light', 'energy'), ('chemical', 'energy'), ('solar', 'energy'),
        ('green', 'plants'), ('thylakoid', 'membranes'), ('cell', 'membrane'),
        ('amino', 'acids'), ('nucleic', 'acids'), ('fatty', 'acids'),
        ('nervous', 'system'), ('solar', 'system'), ('immune', 'system'),
        ('food', 'chain'), ('food', 'web'), ('water', 'cycle'),
        ('greenhouse', 'gases'), ('global', 'warming'), ('climate', 'change'),
        ('machine', 'learning'), ('deep', 'learning'), ('neural', 'network'),
        ('artificial', 'intelligence'), ('natural', 'language'),
        ('data', 'structure'), ('binary', 'tree'), ('linked', 'list'),
        ('operating', 'system'), ('source', 'code'), ('web', 'development'),
        ('quantum', 'mechanics'), ('electric', 'field'), ('magnetic', 'field'),
    ]

    def _merge_compound_keyphrases(self, keyphrases: List[Dict[str, Any]],
                                     original_text: str) -> List[Dict[str, Any]]:
        """Merge single-word keyphrases that form known compounds or appear
        adjacent in the original text.
        
        E.g. if both 'carbon' and 'dioxide' are extracted separately but
        'carbon dioxide' appears in the text, merge them into one keyphrase.
        """
        phrase_set = {kp.get('phrase', '').lower() for kp in keyphrases}
        text_lower = original_text.lower() if original_text else ''
        merged = set()  # track phrases that got merged away
        extra = []  # new compound keyphrases to add

        # Check known compound terms
        for w1, w2 in self.COMPOUND_TERMS:
            compound = f"{w1} {w2}"
            if compound in text_lower and w1 in phrase_set and w2 in phrase_set:
                # Already have the compound? skip
                if compound not in phrase_set:
                    avg_score = 0
                    for kp in keyphrases:
                        if kp.get('phrase', '').lower() in (w1, w2):
                            avg_score = max(avg_score, kp.get('score', 0))
                    extra.append({'phrase': compound, 'score': avg_score + 0.1})
                merged.add(w1)
                merged.add(w2)

        # Also scan for any adjacent single-word keyphrases in text
        single_words = [kp for kp in keyphrases
                        if len(kp.get('phrase', '').split()) == 1
                        and kp.get('phrase', '').lower() not in merged]
        for i, kp1 in enumerate(single_words):
            w1 = kp1['phrase'].lower()
            if w1 in merged:
                continue
            for kp2 in single_words[i+1:]:
                w2 = kp2['phrase'].lower()
                if w2 in merged:
                    continue
                pair = f"{w1} {w2}"
                pair_rev = f"{w2} {w1}"
                if pair in text_lower and pair not in phrase_set:
                    extra.append({'phrase': pair, 'score': max(kp1.get('score',0), kp2.get('score',0)) + 0.05})
                    merged.add(w1)
                    merged.add(w2)
                    break
                elif pair_rev in text_lower and pair_rev not in phrase_set:
                    extra.append({'phrase': pair_rev, 'score': max(kp1.get('score',0), kp2.get('score',0)) + 0.05})
                    merged.add(w1)
                    merged.add(w2)
                    break

        # Filter out merged singles and add compounds
        result = [kp for kp in keyphrases if kp.get('phrase', '').lower() not in merged]
        result.extend(extra)
        # Re-sort by score
        result.sort(key=lambda x: x.get('score', 0), reverse=True)
        return result

    def _clean_keyphrases(self, keyphrases: List[Dict[str, Any]], 
                          main_topic: str) -> List[str]:
        """Clean and filter keyphrases.
        
        For Hindi/Tamil: filters out single-word fragments that are typically
        adjectives, verbs or particles extracted by POS tagging but don't make
        meaningful stand-alone detail nodes (e.g. "मौजूद", "बदलते", "रासायनिक").
        Also caps long phrases to avoid sentence-length detail labels.
        """
        main_topic_lower = main_topic.lower()
        main_words = set(main_topic_lower.split())
        is_non_latin = not main_topic.isascii()
        cleaned = []
        seen = set()
        
        # Sort by score (descending) to get best first
        sorted_kps = sorted(keyphrases, 
                          key=lambda x: x.get("score", 0), 
                          reverse=True)
        
        # Hindi/Tamil: common adjectives, verbs, particles that leak through
        NON_LATIN_NOISE = {
            # Hindi noise words
            'मौजूद', 'बदलते', 'रासायनिक', 'विभिन्न', 'प्रमुख', 'मुख्य',
            'महत्वपूर्ण', 'विशेष', 'सामान्य', 'आवश्यक', 'संभव', 'उचित',
            'नया', 'पुराना', 'बड़ा', 'छोटा', 'अच्छा', 'बुरा', 'पूरा',
            'अधिक', 'कम', 'शामिल', 'संबंधित', 'आधारित', 'निर्भर',
            'स्थित', 'प्राप्त', 'उपलब्ध', 'होने',
            # Tamil noise words
            'உள்ள', 'பல', 'சில', 'முக்கிய', 'புதிய', 'பழைய',
            'பெரிய', 'சிறிய', 'நல்ல', 'மிகவும்', 'தேவையான',
            'அடிப்படை', 'குறிப்பிட்ட', 'மற்ற', 'அனைத்து',
            'உதவுகிறது', 'எடுக்க', 'ஒவ்வொரு', 'நிரல்',
            'தனித்தனியாக', 'விதிமுறைக்கும்', 'எழுதப்படாமலே',
        }
        
        for kp in sorted_kps:
            phrase = kp.get("phrase", "").strip()
            if not phrase or len(phrase) < 2:
                continue
            
            phrase_lower = phrase.lower()
            
            # Skip single-word stopwords that leak through
            if phrase_lower in self.TOPIC_STOPWORDS:
                continue
            
            # Skip generic single words that aren't informative
            if len(phrase_lower.split()) == 1 and phrase_lower in self.GENERIC_SINGLE_WORDS:
                continue
            
            # Skip if exact match with main topic
            if phrase_lower == main_topic_lower:
                continue
            
            # For non-Latin: filter single-word noise and cap phrase length
            if is_non_latin:
                words = phrase.split()
                # Skip single-word fragments that are just adjectives/particles
                if len(words) == 1 and phrase in NON_LATIN_NOISE:
                    continue
                # Skip very short single words (< 3 chars for Devanagari/Tamil)
                if len(words) == 1 and len(phrase) <= 2:
                    continue
                # Cap phrase length: keep at most 3 words to avoid sentence-length labels
                if len(words) > 3:
                    phrase = " ".join(words[:3])
                    phrase_lower = phrase.lower()
            else:
                # For English: also skip subsets of main topic
                if main_topic_lower in phrase_lower or phrase_lower in main_topic_lower:
                    phrase_words = set(phrase_lower.split())
                    overlap = len(main_words & phrase_words)
                    if overlap == len(phrase_words) or overlap == len(main_words):
                        continue
            
            # Skip duplicates
            if phrase_lower in seen:
                continue
            
            seen.add(phrase_lower)
            cleaned.append(phrase)
        
        return cleaned
    
    def _smart_extract_categories(self, keyphrases: List[str], 
                                   original_text: str,
                                   topic_dists: Dict[str, Any]) -> List[str]:
        """Extract meaningful category names using context and semantic analysis"""
        categories = []
        text_lower = original_text.lower()
        
        # Look for key category indicators in text
        category_patterns = [
            # Cause-Effect categories (environmental, analytical texts)
            (["extraction", "harvesting", "farming", "logging", "deforestation", "clearing"], "Causes"),
            (["loss", "erosion", "crisis", "decline", "damage", "destruction"], "Effects"),
            (["agriculture", "soy", "crop", "livestock", "timber", "wood"], "Activities"),
            
            # Technology categories
            (["frontend", "front-end", "client-side", "ui", "user interface"], "Frontend"),
            (["backend", "back-end", "server-side", "server"], "Backend"),
            (["database", "data storage", "sql", "nosql"], "Databases"),
            (["javascript", "js", "scripting"], "JavaScript"),
            (["framework", "library", "tools"], "Frameworks & Tools"),
            (["api", "apis", "service", "endpoint"], "APIs & Services"),
            
            # General categories
            (["technology", "technologies", "tech"], "Technologies"),
            (["concept", "concepts", "principles"], "Core Concepts"),
            (["component", "components", "parts"], "Components"),
            (["feature", "features", "capability"], "Key Features"),
            (["application", "applications", "real-world use"], "Applications"),
            (["benefit", "benefits", "advantage"], "Benefits"),
            
            # Science categories
            (["photosynthesis", "chlorophyll", "chloroplast"], "Photosynthesis"),
            (["energy", "sunlight", "light", "solar"], "Energy"),
            (["plant", "plants", "leaves", "leaf"], "Plant"),
            (["oxygen", "carbon dioxide", "co2", "o2", "gas"], "Chemical Compounds"),
            (["water", "glucose", "sugar", "starch"], "Products"),
            (["cell", "cells", "nucleus", "membrane"], "Cell Biology"),
            (["reaction", "chemical", "equation"], "Chemical Reactions"),
            
            # Hindi general categories
            (["ऊर्जा", "शक्ति", "सूर्य", "प्रकाश"], "ऊर्जा"),
            (["पौधा", "पौधे", "पत्ती", "पत्तियाँ", "वनस्पति"], "पौधे"),
            (["प्रक्रिया", "क्रिया", "विधि"], "प्रक्रिया"),
            (["कारण", "वजह"], "कारण"),
            (["प्रभाव", "परिणाम", "असर"], "प्रभाव"),
            (["उपयोग", "प्रयोग", "इस्तेमाल"], "उपयोग"),
            (["लाभ", "फायदा", "फायदे"], "लाभ"),
            (["प्रकार", "किस्म", "तरह"], "प्रकार"),
            
            # Tamil general categories
            (["ஆற்றல்", "சக்தி", "சூரிய", "ஒளி"], "ஆற்றல்"),
            (["தாவரம்", "தாவரங்கள்", "இலை", "இலைகள்"], "தாவரங்கள்"),
            (["செயல்முறை", "முறை", "வழிமுறை"], "செயல்முறை"),
            (["காரணம்", "காரணங்கள்"], "காரணம்"),
            (["விளைவு", "விளைவுகள்", "பாதிப்பு"], "விளைவு"),
            (["பயன்", "பயன்பாடு", "பயன்கள்"], "பயன்பாடு"),
            (["நன்மை", "நன்மைகள்", "பலன்"], "நன்மைகள்"),
        ]
        
        # Find categories mentioned in text (but limit to max 5)
        max_categories = min(5, max(3, len(keyphrases) // 3))
        for patterns, category_name in category_patterns:
            if len(categories) >= max_categories:
                break
            if any(pattern in text_lower for pattern in patterns):
                if category_name not in categories:
                    categories.append(category_name)
        
        # Use high-scoring keyphrases as categories if they're broad enough
        # But avoid duplicates by checking similarity
        
        # Check if text is non-Latin (Hindi/Tamil)
        is_non_latin = original_text and not original_text[:20].isascii()
        
        for phrase in keyphrases[:12]:
            phrase_words = phrase.lower().split()
            # Keep category names short: max 2 words for all languages
            max_cat_words = 2
            if len(phrase_words) <= max_cat_words and len(categories) < max_categories:
                if is_non_latin:
                    capitalized = phrase  # Don't capitalize non-Latin scripts
                else:
                    capitalized = " ".join(word.capitalize() for word in phrase_words)
                # Check if not too similar to existing categories
                is_duplicate = False
                for existing_cat in categories:
                    existing_lower = existing_cat.lower()
                    capitalized_lower = capitalized.lower()
                    if capitalized_lower in existing_lower or existing_lower in capitalized_lower:
                        is_duplicate = True
                        break
                    # Check word overlap
                    existing_words = set(existing_lower.split())
                    cap_words = set(capitalized_lower.split())
                    if existing_words and cap_words:
                        overlap = len(existing_words & cap_words)
                        if overlap > 0 and overlap / max(len(existing_words), len(cap_words)) > 0.7:
                            is_duplicate = True
                            break
                if not is_duplicate and capitalized not in categories:
                    categories.append(capitalized)
        
        # Use topic modeling results if available
        topics = topic_dists.get("topics", [])
        for topic in topics[:4]:
            if isinstance(topic, dict):
                words = topic.get("words", [])
                if words and len(categories) < max_categories:
                    cat_name = " & ".join(words[:2]).title()
                    if cat_name not in categories:
                        categories.append(cat_name)
        
        # Only add defaults if we have very few categories
        if len(categories) < 2:
            lang = self._detect_language(original_text)
            if lang == "hi":
                defaults = ["मुख्य अवधारणाएँ", "विवरण"]
            elif lang == "ta":
                defaults = ["முக்கிய கருத்துகள்", "விவரங்கள்"]
            else:
                defaults = ["Key Concepts", "Details"]
            for default in defaults:
                if len(categories) >= 2:
                    break
                if default not in categories:
                    categories.append(default)
        
        return categories[:max_categories]
    
    def _smart_map_details(self, keyphrases: List[str], 
                           categories: List[str],
                           original_text: str) -> Dict[str, List[str]]:
        """Map keyphrases to categories using semantic matching"""
        details_map = {cat: [] for cat in categories}
        text_lower = original_text.lower()
        used_phrases = set()
        
        # Define semantic associations for better mapping
        category_keywords = {
            # Environmental/Causal categories
            "causes": ["extraction", "harvesting", "farming", "logging", "illegal", "deforestation", "clearing", "cutting"],
            "effects": ["loss", "erosion", "decline", "damage", "destruction", "impact", "crisis", "biodiversity"],
            "activities": ["agriculture", "soy", "crop", "livestock", "timber", "wood", "land", "large-scale"],
            
            # Technology categories
            "frontend": ["html", "css", "react", "angular", "vue", "ui", "design", "style", "visual", "page", "structure", "content"],
            "backend": ["node", "server", "api", "database", "logic", "python", "java", "express", "django"],
            "databases": ["sql", "nosql", "mongodb", "postgresql", "mysql", "data", "storage", "query"],
            "javascript": ["js", "dynamic", "interactive", "behavior", "script", "function"],
            "frameworks": ["react", "angular", "vue", "express", "django", "framework", "library"],
            "technologies": ["technology", "tool", "tech", "platform"],
            "apis": ["api", "endpoint", "service", "rest", "graphql"],
            
            # Science categories
            "energy": ["sunlight", "light", "solar", "atp", "chemical energy", "heat", "radiation"],
            "plant": ["leaves", "roots", "stem", "chloroplast", "cell", "stomata", "trees"],
            "chemical compounds": ["oxygen", "carbon dioxide", "water", "glucose", "co2", "o2", "h2o"],
            "products": ["glucose", "sugar", "starch", "food", "organic"],
            "chemical reactions": ["reaction", "equation", "formula", "catalyst", "enzyme"],
            "photosynthesis": ["chlorophyll", "pigment", "absorption", "spectrum"],
            
            # Hindi semantic categories
            "ऊर्जा": ["सूर्य", "प्रकाश", "शक्ति", "ताप", "विकिरण", "सौर"],
            "पौधे": ["पत्ती", "पत्तियाँ", "जड़", "तना", "वनस्पति", "पेड़"],
            "प्रक्रिया": ["क्रिया", "विधि", "चरण", "अवस्था", "रूपांतरण"],
            "कारण": ["वजह", "कारक", "मूल"],
            "प्रभाव": ["परिणाम", "असर", "नतीजा", "प्रतिक्रिया"],
            "उपयोग": ["प्रयोग", "इस्तेमाल", "काम"],
            "लाभ": ["फायदा", "फायदे", "गुण"],
            "मुख्य अवधारणाएँ": ["परिभाषा", "अर्थ", "सिद्धांत", "नियम"],
            "विवरण": ["जानकारी", "तथ्य", "विशेषता"],
            
            # Tamil semantic categories
            "ஆற்றல்": ["சூரிய", "ஒளி", "சக்தி", "வெப்பம்"],
            "தாவரங்கள்": ["இலை", "இலைகள்", "வேர்", "தண்டு", "மரம்"],
            "செயல்முறை": ["முறை", "வழிமுறை", "நிலை", "படிநிலை"],
            "காரணம்": ["காரணி", "மூலம்"],
            "விளைவு": ["பாதிப்பு", "மாற்றம்", "எதிர்வினை"],
            "பயன்பாடு": ["பயன்", "பயன்கள்", "உபயோகம்"],
            "நன்மைகள்": ["பலன்", "சிறப்பு"],
            "முக்கிய கருத்துக்கள்": ["வரையறை", "பொருள்", "கோட்பாடு"],
        }
        
        # First pass: match keyphrases to categories by semantic relevance
        for phrase in keyphrases:
            phrase_lower = phrase.lower()
            best_category = None
            best_score = 0
            
            for category in categories:
                cat_lower = category.lower()
                score = 0
                
                # Direct mention in category
                if phrase_lower in cat_lower or cat_lower in phrase_lower:
                    score += 5
                
                # Check semantic keywords
                for key_part in cat_lower.split():
                    if key_part in category_keywords:
                        keywords = category_keywords[key_part]
                        if any(kw in phrase_lower for kw in keywords):
                            score += 3
                
                # Check proximity in original text
                cat_words = cat_lower.split()
                phrase_words = phrase_lower.split()
                for cat_word in cat_words:
                    for phrase_word in phrase_words:
                        # Find both in text and check distance
                        if cat_word in text_lower and phrase_word in text_lower:
                            cat_pos = text_lower.find(cat_word)
                            phrase_pos = text_lower.find(phrase_word)
                            if abs(cat_pos - phrase_pos) < 100:  # Within 100 chars
                                score += 2
                
                # Dynamic limit per category — cap at 4 to keep the map readable
                max_per_category = min(4, max(2, len(keyphrases) // len(categories))) if categories else 4
                
                if score > best_score and len(details_map[category]) < max_per_category:
                    best_score = score
                    best_category = category
            
            # Assign to best matching category
            if best_category and best_score > 0:
                details_map[best_category].append(phrase)
                used_phrases.add(phrase_lower)
        
        # Second pass: distribute remaining phrases evenly (cap at 4 per category)
        remaining = [kp for kp in keyphrases if kp.lower() not in used_phrases]
        max_per_category = min(4, max(3, len(keyphrases) // len(categories))) if categories else 4
        for i, phrase in enumerate(remaining):
            # Find category with fewest items
            min_category = min(categories, key=lambda c: len(details_map[c]))
            if len(details_map[min_category]) < max_per_category:
                details_map[min_category].append(phrase)
        
        return details_map
    
    # Default categories per language
    DEFAULT_CATEGORIES = {
        "en": ["Overview", "Key Concepts", "Applications", "Related Topics"],
        "hi": ["अवलोकन", "मुख्य अवधारणाएँ", "अनुप्रयोग", "संबंधित विषय"],
        "ta": ["கண்ணோட்டம்", "முக்கிய கருத்துகள்", "பயன்பாடுகள்", "தொடர்புடைய தலைப்புகள்"],
    }
    
    def _get_default_categories(self, text: str) -> List[str]:
        """Return default category names in the appropriate language"""
        lang = self._detect_language(text)
        return list(self.DEFAULT_CATEGORIES.get(lang, self.DEFAULT_CATEGORIES["en"]))
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on script"""
        import re
        if not text:
            return "en"
        devanagari = len(re.findall(r'[\u0900-\u097F]', text))
        tamil = len(re.findall(r'[\u0B80-\u0BFF]', text))
        latin = len(re.findall(r'[a-zA-Z]', text))
        total = devanagari + tamil + latin
        if total == 0:
            return "en"
        if devanagari / total > 0.3:
            return "hi"
        if tamil / total > 0.3:
            return "ta"
        return "en"

    # Comprehensive stopwords for main topic extraction (multilingual)
    TOPIC_STOPWORDS = {
        # English
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'and', 'but', 'or',
        'nor', 'not', 'so', 'yet', 'of', 'to', 'in', 'for', 'on', 'with',
        'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
        'after', 'above', 'below', 'between', 'about', 'against', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there',
        'when', 'where', 'why', 'how', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'it', 'its', 'also', 'just',
        'only', 'very', 'too', 'much', 'many', 'more', 'most', 'other',
        'some', 'any', 'all', 'each', 'every', 'both', 'few', 'than',
        'such', 'no', 'up', 'out', 'if', 'because', 'while', 'until',
        'consists', 'they', 'them', 'their', 'we', 'our', 'you', 'your',
        # French
        'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'est', 'sont',
        'et', 'ou', 'en', 'au', 'aux', 'ce', 'ces', 'qui', 'que', 'par',
        'sur', 'dans', 'pour', 'avec', 'sans', 'sous', 'entre', 'vers',
        'se', 'sa', 'son', 'ses', 'nous', 'vous', 'ils', 'elles', 'leur',
        'ne', 'pas', 'plus', 'très', 'bien', 'aussi', 'comme', 'mais',
        # Hindi
        'का', 'के', 'की', 'है', 'में', 'को', 'से', 'और', 'एक',
        'पर', 'ने', 'यह', 'वह', 'इस', 'उस', 'नहीं', 'था', 'थी',
        'थे', 'हैं', 'भी', 'कि', 'जो', 'तो', 'हो', 'कर', 'या',
        'अपने', 'अपनी', 'अपना', 'लिए', 'कुछ', 'साथ', 'बाद',
        'पहले', 'दो', 'बहुत', 'अब', 'जब', 'तक', 'उन', 'इन',
        'हम', 'मैं', 'तुम', 'आप', 'वे', 'ये', 'होता', 'होती',
        'होते', 'रहा', 'रही', 'रहे', 'गया', 'गई', 'गए',
        'सकता', 'सकती', 'सकते', 'करता', 'करती', 'करते',
        'हुआ', 'हुई', 'हुए', 'ऐसे', 'कैसे', 'जैसे',
        'बस', 'फिर', 'अगर', 'मगर', 'लेकिन', 'क्योंकि',
        'इसलिए', 'वाला', 'वाली', 'वाले', 'सब', 'कोई',
        'जिसे', 'जिसमें', 'जिसका', 'जिसकी', 'जिसके',
        'कहा', 'कहते', 'जाता', 'जाती', 'द्वारा', 'रूप',
        # Tamil
        'ஒரு', 'இந்த', 'ஆகும்', 'அந்த', 'என்று', 'என்ற',
        'இது', 'அது', 'மற்றும்', 'என', 'உள்ள', 'கொண்ட',
        'போது', 'அவர்', 'இருந்து', 'செய்து', 'வரும்', 'பின்',
        'மேலும்', 'தான்', 'அவன்', 'அவள்', 'நான்', 'நாம்',
        'நீ', 'நீங்கள்', 'அவர்கள்', 'இருக்கும்', 'இல்லை',
        'உள்ளது', 'என்பது', 'பற்றி', 'அதன்', 'இதன்',
        'ஆகிய', 'முதல்', 'வரை', 'ஆனால்', 'எனவே',
        'ஏனெனில்', 'அல்லது', 'போன்ற', 'கொண்டு', 'வந்து',
        'சென்று', 'செய்யும்', 'இருந்தது',
    }
    
    # Common adjectives/descriptors to filter from main topic
    TOPIC_ADJECTIVES = {
        'green', 'large', 'small', 'big', 'new', 'old', 'first', 'last',
        'good', 'bad', 'great', 'important', 'major', 'minor', 'key',
        'main', 'primary', 'secondary', 'general', 'specific', 'common',
        'various', 'different', 'certain', 'particular', 'simple', 'complex',
        'basic', 'advanced', 'modern', 'ancient', 'natural', 'artificial',
        'real', 'actual', 'effective', 'significant', 'essential', 'vital',
        'critical', 'fundamental', 'central', 'core', 'entire', 'whole',
        'complete', 'full', 'total', 'overall', 'international', 'national',
        'local', 'global', 'several', 'numerous', 'multiple', 'single',
        'process', 'called', 'known', 'used', 'defined', 'considered',
        'chemical', 'physical', 'organic', 'inorganic', 'molecular', 'atomic',
        'scientific', 'technical', 'environmental', 'ecological', 'industrial',
        # Broad nouns that should not be in a topic title
        'biological', 'plants', 'plant', 'animals', 'animal', 'organisms',
        'cells', 'cell', 'molecules', 'molecule', 'systems', 'system',
        'components', 'component', 'elements', 'element',
        'structure', 'structures', 'function', 'functions',
        'types', 'type', 'forms', 'form', 'kind', 'kinds',
        'things', 'thing', 'way', 'ways', 'part', 'parts',
        'role', 'roles', 'plays', 'plays', 'involves', 'involves',
        'occurs', 'takes', 'place', 'found', 'convert', 'converts',
        'produce', 'produces', 'makes', 'make', 'uses', 'using',
        'requires', 'require', 'needs', 'need', 'help', 'helps',
        'include', 'includes', 'including', 'contain', 'contains',
        'sunlight', 'light', 'energy', 'water',
        # Hindi broad/generic words to filter from main topic
        'जैविक', 'प्रक्रिया', 'प्रमुख', 'मुख्य', 'विभिन्न', 'महत्वपूर्ण',
        'पौधे', 'पौधा', 'पेड़', 'सूर्य', 'रोशनी', 'ऊर्जा', 'पानी',
        'जल', 'कोशिका', 'कोशिकाओं', 'रासायनिक', 'प्राकृतिक',
        # Tamil broad/generic words to filter from main topic
        'தாவரங்கள்', 'தாவரம்', 'சூரிய', 'சூரியன்', 'ஒளி', 'ஆற்றல்',
        'நீர்', 'தண்ணீர்', 'உயிரியல்', 'செல்', 'செல்கள்', 'இயற்கை',
    }

    def _extract_main_topic(self, text: str) -> str:
        """Extract main topic from first sentence intelligently.
        
        Aims for 1-2 strong content words (e.g. 'Photosynthesis') rather than
        a long phrase. For non-Latin scripts allows up to 3 words.
        """
        if not text:
            return "Main Topic"
        
        import re
        # Split by common sentence delimiters (including Hindi danda ।)
        sentences = re.split(r'[।\.\!\?\n]+', text)
        first = sentences[0].strip() if sentences else text
        
        # Determine language
        is_non_latin = not first[:20].isascii() if first else False
        
        # Extract meaningful words
        words = []
        skip_set = self.TOPIC_STOPWORDS | self.TOPIC_ADJECTIVES
        
        for w in first.split():
            w_clean = w.strip('.,!?;:।')
            if not w_clean or len(w_clean) < 2:
                continue
            
            if w_clean.lower() in skip_set:
                continue
            
            # For non-Latin: don't capitalize (no case concept)
            if w_clean.isascii():
                words.append(w_clean.capitalize())
            else:
                words.append(w_clean)
            
            # For non-Latin allow up to 3 words; for English prefer 1-2
            max_words = 3 if is_non_latin else 2
            if len(words) >= max_words:
                break
        
        return " ".join(words) if words else "Main Topic"
    
    def _build_hierarchy(self, main_topic: str, categories: List[str], 
                         details_map: Dict[str, List[str]],
                         relations: List[Dict[str, Any]] = None) -> None:
        """
        Build hierarchical graph structure with meaningful relations
        
        Structure:
        - center: main topic
        - cat_0, cat_1, ...: category nodes (level 1)
        - det_0_0, det_0_1, ...: detail nodes under categories (level 2)
        - edges with context-aware relation labels
        """
        if relations is None:
            relations = []
        
        # Level 0: Center
        self.graph.add_node("center", label=main_topic, nodeType="main", level=0)
        
        # Level 1: Categories
        for i, cat in enumerate(categories):
            cat_id = f"cat_{i}"
            self.graph.add_node(cat_id, label=cat, nodeType="category", level=1)
            
            # Connect to main topic (no label for cleaner look)
            self.graph.add_edge("center", cat_id, label="", relation="")
            
            # Level 2: Details (deduplicated across all categories)
            details = details_map.get(cat, [])
            seen_detail_labels = set()  # Track unique detail labels
            
            for j, det in enumerate(details):
                det_lower = det.lower()
                
                # Skip if this label already exists globally
                all_existing_labels = {self.graph.nodes[n].get("label", "").lower() 
                                      for n in self.graph.nodes() if n != "center"}
                if det_lower in all_existing_labels or det_lower in seen_detail_labels:
                    continue
                    
                seen_detail_labels.add(det_lower)
                det_id = f"det_{i}_{j}"
                self.graph.add_node(det_id, label=det, nodeType="detail", level=2)
                
                # Connect to category (no label for cleaner look)
                self.graph.add_edge(cat_id, det_id, label="", relation="")
    
    def _get_category_edge_label(self, category: str) -> str:
        """Get meaningful edge label for category connection"""
        # Always return empty for cleaner look
        return ""
    
    def _get_detail_edge_label(self, category: str, detail: str, rel_type: str) -> str:
        """Get meaningful edge label for detail connection"""
        cat_lower = category.lower()
        det_lower = detail.lower()
        
        # Technology-specific labels
        if "frontend" in cat_lower:
            if "html" in det_lower:
                return "structures"
            elif "css" in det_lower:
                return "styles"
            elif "react" in det_lower or "framework" in det_lower:
                return "builds with"
        elif "backend" in cat_lower:
            if "node" in det_lower:
                return "powered by"
            elif "api" in det_lower:
                return "provides"
            elif "server" in det_lower:
                return "runs on"
        elif "database" in cat_lower:
            if "sql" in det_lower:
                return "uses"
            elif "nosql" in det_lower:
                return "stores in"
        elif "javascript" in cat_lower:
            if "dynamic" in det_lower or "interactive" in det_lower:
                return "enables"
        
        # Generic relation-based labels
        if rel_type == "IS_A":
            return "is a"
        elif rel_type == "PART_OF":
            return "part of"
        elif rel_type == "USES":
            return "uses"
        elif rel_type == "REQUIRES":
            return "requires"
        elif rel_type == "CAUSES":
            return "leads to"
        
        return ""  # Empty for less important connections
    
    def _get_relation_type(self, concept1: str, concept2: str) -> str:
        """Get relation type between two concepts from relation_map"""
        key = (concept1, concept2)
        reverse_key = (concept2, concept1)
        
        if key in self.relation_map:
            return self.relation_map[key]
        elif reverse_key in self.relation_map:
            return self.relation_map[reverse_key]
        else:
            # Infer relation type from context
            if any(word in concept2 for word in ["framework", "library", "tool"]):
                return "USES"
            elif any(word in concept2 for word in ["api", "service"]):
                return "PROVIDES"
            else:
                return ""  # Empty instead of generic "RELATES_TO"
    
    def _calculate_layout(self) -> Dict[str, Dict[str, float]]:
        """Calculate hierarchical tree layout (top-down)"""
        positions = {}
        
        # Canvas dimensions (SVG viewBox 0 0 3200 900)
        canvas_width = 3200  # Increased for maximum horizontal space
        canvas_height = 900
        
        # Level spacing
        level_0_y = 100  # Main topic at top
        level_1_y = 300  # Categories in middle
        level_2_y = 600  # Details at bottom
        
        # Center node (main topic)
        positions["center"] = {"x": canvas_width / 2, "y": level_0_y}
        
        # Get categories
        categories = [n for n in self.graph.nodes() 
                     if self.graph.nodes[n].get("level") == 1]
        n_cats = len(categories)
        
        if n_cats == 0:
            return positions
        
        # Calculate horizontal spacing for categories with generous padding
        cat_spacing = canvas_width / (n_cats + 1)
        
        # Collect all detail nodes for global positioning
        all_details = []
        cat_details_map = {}
        
        for cat_id in categories:
            details = list(self.graph.successors(cat_id))
            cat_details_map[cat_id] = details
            all_details.extend(details)
        
        total_details = len(all_details)
        
        # Place categories horizontally
        for i, cat_id in enumerate(categories):
            cat_x = cat_spacing * (i + 1)
            positions[cat_id] = {"x": cat_x, "y": level_1_y}
        
        # Place all details in a single row with even spacing to prevent overlap
        if total_details > 0:
            # Use full canvas width with equal spacing
            detail_spacing = canvas_width / (total_details + 1)
            detail_idx = 0
            
            for cat_id in categories:
                details = cat_details_map.get(cat_id, [])
                for det_id in details:
                    det_x = detail_spacing * (detail_idx + 1)
                    positions[det_id] = {"x": det_x, "y": level_2_y}
                    detail_idx += 1
        
        return positions
    
    def _build_output(self, layout: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Build output for frontend with enhanced edge information"""
        nodes = []
        edges = []
        
        # Build nodes with proper styling
        for node_id in self.graph.nodes():
            data = self.graph.nodes[node_id]
            pos = layout.get(node_id, {"x": 800, "y": 450})
            level = data.get("level", 2)
            
            # Determine node appearance based on level
            if level == 0:
                node_type = "main"
                size = 80
            elif level == 1:
                node_type = "category"
                size = 60
            else:
                node_type = "detail"
                size = 45
            
            nodes.append({
                "id": node_id,
                "label": data.get("label", node_id),
                "type": node_type,
                "nodeType": data.get("nodeType", "detail"),
                "x": pos["x"],
                "y": pos["y"],
                "size": size,
                "level": level
            })
        
        # Build edges with relation information (deduplicated)
        seen_edges = set()
        for src, tgt, data in self.graph.edges(data=True):
            edge_key = f"{src}_{tgt}"
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            
            edges.append({
                "id": edge_key,
                "from": src,
                "to": tgt,
                "source": src,
                "target": tgt,
                "label": data.get("label", ""),
                "relation": data.get("relation", "")
            })
        
        return {"nodes": nodes, "edges": edges}
