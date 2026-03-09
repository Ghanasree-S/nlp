"""Quick test for mindmap keyword quality"""
from nlp.preprocessing.preprocessor import TextPreprocessor
from nlp.keyphrase.extractor import KeyphraseExtractor
from mindmap_gen.mindmap_generator import MindMapGenerator
from nlp.topic_model.topic_modeler import TopicModeler
from nlp.relation.relation_extractor import RelationExtractor

pp = TextPreprocessor()
ke = KeyphraseExtractor()
tm = TopicModeler()
re2 = RelationExtractor()
mg = MindMapGenerator()

text = """Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. Supervised learning uses labeled training data to make predictions. Unsupervised learning discovers hidden patterns in unlabeled data. Neural networks are composed of layers of interconnected nodes. Deep learning uses multiple layers to progressively extract higher-level features. Common algorithms include decision trees, random forests, and support vector machines."""

preprocessed = pp.process(text)
keyphrases = ke.extract(preprocessed, top_k=20)
print("=== KEYPHRASES ===")
for kp in keyphrases:
    print(f"  {kp['phrase']:30s} score={kp.get('score',0):.3f}")

topics = tm.model_topics(preprocessed, keyphrases)
topics["original_text"] = preprocessed.get("original_text", "")
relations = re2.extract(preprocessed, keyphrases)
mindmap = mg.generate(keyphrases, topics, relations)
print()
print("=== MINDMAP NODES ===")
for node in mindmap["graph"]["nodes"]:
    indent = "  " * node.get("level", 0)
    print(f"{indent}[L{node.get('level',0)}] {node['label']}")
