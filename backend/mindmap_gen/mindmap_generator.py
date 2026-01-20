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
            return ["Overview", "Key Concepts", "Applications", "Related Topics"], {}
        
        # Get topic distribution and original text
        topic_dists = topics.get("topic_distribution", {})
        original_text = topics.get("original_text", "")
        
        # Clean and score keyphrases
        cleaned_kps = self._clean_keyphrases(keyphrases, main_topic)
        
        if not cleaned_kps:
            return ["Overview", "Key Concepts", "Applications", "Related Topics"], {}
        
        # Smart category extraction from context
        categories = self._smart_extract_categories(cleaned_kps, original_text, topic_dists)
        
        # Smart mapping of details to categories
        details_map = self._smart_map_details(cleaned_kps, categories, original_text)
        
        return categories, details_map
    
    def _clean_keyphrases(self, keyphrases: List[Dict[str, Any]], 
                          main_topic: str) -> List[str]:
        """Clean and filter keyphrases"""
        main_topic_lower = main_topic.lower()
        main_words = set(main_topic_lower.split())
        cleaned = []
        seen = set()
        
        # Sort by score (descending) to get best first
        sorted_kps = sorted(keyphrases, 
                          key=lambda x: x.get("score", 0), 
                          reverse=True)
        
        for kp in sorted_kps:
            phrase = kp.get("phrase", "").strip()
            if not phrase or len(phrase) < 2:
                continue
            
            phrase_lower = phrase.lower()
            
            # Skip if exact match with main topic
            if phrase_lower == main_topic_lower:
                continue
            
            # Skip if phrase is a subset of main topic or vice versa
            if main_topic_lower in phrase_lower or phrase_lower in main_topic_lower:
                # Exception: keep if it's a completely different phrase with one word overlap
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
            (["application", "applications", "use"], "Applications"),
            (["benefit", "benefits", "advantage"], "Benefits"),
        ]
        
        # Find categories mentioned in text
        for patterns, category_name in category_patterns:
            if any(pattern in text_lower for pattern in patterns):
                if category_name not in categories:
                    categories.append(category_name)
        
        # Use high-scoring keyphrases as categories if they're broad enough
        # But avoid duplicates by checking similarity
        # Dynamic limit based on keyphrases count
        max_categories = min(8, max(3, len(keyphrases) // 3))
        
        for phrase in keyphrases[:12]:
            phrase_words = phrase.lower().split()
            # Only use 1-2 word phrases as categories
            if len(phrase_words) <= 2 and len(categories) < max_categories:
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
            defaults = ["Key Concepts", "Details"]
            for default in defaults:
                if len(categories) >= 2:
                    break
                if default not in categories:
                    categories.append(default)
        
        return categories  # Return all found categories (no hard limit)
    
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
                
                # Dynamic limit per category based on total keyphrases
                max_per_category = max(2, len(keyphrases) // len(categories)) if categories else 5
                
                if score > best_score and len(details_map[category]) < max_per_category:
                    best_score = score
                    best_category = category
            
            # Assign to best matching category
            if best_category and best_score > 0:
                details_map[best_category].append(phrase)
                used_phrases.add(phrase_lower)
        
        # Second pass: distribute remaining phrases evenly
        remaining = [kp for kp in keyphrases if kp.lower() not in used_phrases]
        max_per_category = max(3, len(keyphrases) // len(categories)) if categories else 5
        for i, phrase in enumerate(remaining):
            # Find category with fewest items
            min_category = min(categories, key=lambda c: len(details_map[c]))
            if len(details_map[min_category]) < max_per_category:
                details_map[min_category].append(phrase)
        
        return details_map
    
    def _extract_main_topic(self, text: str) -> str:
        """Extract main topic from first sentence intelligently"""
        if not text:
            return "Main Topic"
        
        # Get first sentence
        sentences = text.split('.')
        first = sentences[0] if sentences else text
        
        # Common stopwords
        stops = {'the', 'a', 'an', 'is', 'are', 'and', 'or', 'of', 'to', 'in', 
                'for', 'with', 'by', 'on', 'at', 'consists', 'that', 'this'}
        
        # Extract meaningful words
        words = []
        for w in first.split():
            w_clean = w.strip('.,!?;:')
            if w_clean.lower() not in stops and len(w_clean) > 2:
                words.append(w_clean.capitalize())
                if len(words) >= 2:
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
