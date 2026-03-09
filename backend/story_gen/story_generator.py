"""
Story Generator — Keyword-to-Story using NLP techniques

NLP Techniques Used:
1. N-gram Language Model (Markov Chain) trained on fairy tales corpus
2. POS Tagging (SpaCy) to understand keyword roles
3. Sentence segmentation and coherent text generation
4. Keyword seeding — stories are biased toward user-provided words

Pipeline:
  keywords → POS-tag → build seed sentences → Markov chain expansion
  → narrative arc structuring → final story
"""

import os
import re
import random
import pickle
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

CORPUS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "..", "dataset", "cleaned_merged_fairy_tales_without_eos.txt"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "markov_model.pkl")


class StoryGenerator:
    """
    Generates short stories from keywords using an N-gram Markov Chain
    trained on a fairy tales corpus, with SpaCy-based keyword integration.
    """

    def __init__(self):
        self.bigrams: Dict[str, List[str]] = {}
        self.trigrams: Dict[str, List[str]] = {}
        self.sentence_starters: List[str] = []
        self.corpus_sentences: List[str] = []
        self._trained = False
        self._load_or_train()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _load_or_train(self):
        """Load cached model or train from corpus."""
        if os.path.exists(MODEL_PATH):
            try:
                with open(MODEL_PATH, "rb") as f:
                    data = pickle.load(f)
                self.bigrams = data["bigrams"]
                self.trigrams = data["trigrams"]
                self.sentence_starters = data["starters"]
                self.corpus_sentences = data.get("sentences", [])
                self._trained = True
                logger.info("Loaded cached Markov model (%d bigram keys)", len(self.bigrams))
                return
            except Exception as e:
                logger.warning("Failed to load cached model: %s", e)

        if not os.path.exists(CORPUS_PATH):
            logger.warning("Fairy tales corpus not found at %s", CORPUS_PATH)
            return

        logger.info("Training Markov model on fairy tales corpus...")
        self._train_from_corpus()

    def _train_from_corpus(self):
        """Build bigram + trigram tables from the fairy tales corpus."""
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            text = f.read()

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Keep only well-formed sentences (5-80 words)
        sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 80]

        # Store a random sample of sentences for keyword matching
        self.corpus_sentences = random.sample(sentences, min(50000, len(sentences)))

        bigrams: Dict[str, list] = defaultdict(list)
        trigrams: Dict[str, list] = defaultdict(list)
        starters: List[str] = []

        for sent in sentences:
            words = sent.split()
            if len(words) < 3:
                continue

            # First two words are potential starters
            starters.append(words[0] + " " + words[1])

            # Build bigram table
            for i in range(len(words) - 1):
                bigrams[words[i].lower()].append(words[i + 1])

            # Build trigram table
            for i in range(len(words) - 2):
                key = words[i].lower() + " " + words[i + 1].lower()
                trigrams[key].append(words[i + 2])

        self.bigrams = dict(bigrams)
        self.trigrams = dict(trigrams)
        self.sentence_starters = starters
        self._trained = True

        # Cache to disk
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump({
                    "bigrams": self.bigrams,
                    "trigrams": self.trigrams,
                    "starters": self.sentence_starters,
                    "sentences": self.corpus_sentences,
                }, f)
            logger.info("Markov model cached (%d bigram keys, %d trigram keys)",
                        len(self.bigrams), len(self.trigrams))
        except Exception as e:
            logger.warning("Could not cache model: %s", e)

    # ------------------------------------------------------------------
    # Story generation
    # ------------------------------------------------------------------
    def generate(self, keywords: List[str], num_sentences: int = 12,
                 language: str = "en") -> Dict[str, Any]:
        """
        Generate a story from keywords.

        Args:
            keywords: List of words/phrases from the user
            num_sentences: Target number of sentences (default 12)
            language: Language code

        Returns:
            Dict with title, summary, story (full text), and panels (for comic view)
        """
        if not self._trained:
            return self._fallback_story(keywords)

        # Classify keywords by POS using SpaCy
        keyword_roles = self._classify_keywords(keywords)

        # Generate story sentences
        story_sentences = self._build_story(keywords, keyword_roles, num_sentences)

        # Build title from keywords
        title = self._build_title(keywords, keyword_roles)

        # Build summary
        summary = story_sentences[0] if story_sentences else "A tale woven from your words."

        # Split into comic panels
        panels = self._story_to_panels(story_sentences, keywords)

        full_story = " ".join(story_sentences)

        return {
            "title": title,
            "summary": summary,
            "story": full_story,
            "panels": panels,
            "keywords_used": keywords,
            "nlp_info": {
                "technique": "N-gram Markov Chain Language Model",
                "corpus": "Fairy Tales (3.7M words)",
                "keyword_roles": keyword_roles,
            }
        }

    def _classify_keywords(self, keywords: List[str]) -> Dict[str, str]:
        """Use SpaCy POS tagging to classify each keyword."""
        roles = {}
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            for kw in keywords:
                doc = nlp(kw)
                if doc:
                    # Get the dominant POS
                    pos_tags = [token.pos_ for token in doc if token.pos_ not in ("PUNCT", "SPACE", "DET")]
                    if pos_tags:
                        main_pos = max(set(pos_tags), key=pos_tags.count)
                        roles[kw] = main_pos
                    else:
                        roles[kw] = "NOUN"
                else:
                    roles[kw] = "NOUN"
        except Exception:
            for kw in keywords:
                roles[kw] = "NOUN"
        return roles

    def _build_story(self, keywords: List[str], keyword_roles: Dict[str, str],
                     target_sentences: int) -> List[str]:
        """Build story sentences weaving in keywords."""
        sentences = []
        used_keywords = set()

        # Phase 1: Opening — find corpus sentences containing a keyword
        for kw in keywords[:3]:
            seed_sent = self._find_seed_sentence(kw)
            if seed_sent:
                sentences.append(seed_sent)
                used_keywords.add(kw)
                if len(sentences) >= 3:
                    break

        # If no seed sentences found, generate opening with Markov chain
        if not sentences:
            opening = self._generate_markov_sentence(keywords[0] if keywords else None, max_words=25)
            sentences.append(opening)

        # Phase 2: Middle — Markov chain generation seeded by keywords
        remaining_kws = [kw for kw in keywords if kw not in used_keywords]
        while len(sentences) < target_sentences - 2:
            # Alternate between keyword-seeded and free generation
            if remaining_kws and random.random() < 0.6:
                kw = remaining_kws.pop(0)
                sent = self._generate_markov_sentence(kw, max_words=25)
                used_keywords.add(kw)
            else:
                # Free generation from a random keyword already used
                seed = random.choice(keywords) if keywords else None
                sent = self._generate_markov_sentence(seed, max_words=25)
            sentences.append(sent)

        # Phase 3: Closing — generate a concluding sentence
        closing_starters = [
            "And so,", "In the end,", "Finally,", "From that day on,",
            "Thus,", "And they all", "At last,"
        ]
        last_kw = keywords[-1] if keywords else keywords[0] if keywords else ""
        closing = random.choice(closing_starters) + " " + self._generate_markov_sentence(last_kw, max_words=20)
        sentences.append(closing)

        return sentences

    def _find_seed_sentence(self, keyword: str) -> Optional[str]:
        """Find a sentence from the corpus containing the keyword."""
        kw_lower = keyword.lower()
        matches = [s for s in self.corpus_sentences if kw_lower in s.lower()]
        if matches:
            # Pick a short-ish one
            good = [s for s in matches if len(s.split()) <= 30]
            return random.choice(good) if good else random.choice(matches[:10])
        return None

    def _generate_markov_sentence(self, seed_word: Optional[str] = None,
                                   max_words: int = 25) -> str:
        """Generate a sentence using the Markov chain, optionally seeded."""
        words = []

        # Try to start from the seed word
        if seed_word and seed_word.lower() in self.bigrams:
            words.append(seed_word.capitalize())
            next_word = random.choice(self.bigrams[seed_word.lower()])
            words.append(next_word)
        elif self.sentence_starters:
            starter = random.choice(self.sentence_starters)
            words.extend(starter.split())
        else:
            return f"Once upon a time, there was a tale about {seed_word or 'wonder'}."

        # Extend using trigrams (preferred) then bigrams
        for _ in range(max_words - len(words)):
            # Try trigram first
            if len(words) >= 2:
                tri_key = words[-2].lower() + " " + words[-1].lower()
                if tri_key in self.trigrams:
                    next_word = random.choice(self.trigrams[tri_key])
                    words.append(next_word)
                    # Stop at sentence-ending punctuation
                    if next_word.endswith(('.', '!', '?')):
                        break
                    continue

            # Fall back to bigram
            bi_key = words[-1].lower().rstrip('.!?,;:')
            if bi_key in self.bigrams:
                next_word = random.choice(self.bigrams[bi_key])
                words.append(next_word)
                if next_word.endswith(('.', '!', '?')):
                    break
            else:
                break

        text = " ".join(words)
        # Ensure it ends with punctuation
        if not text.endswith(('.', '!', '?')):
            text += "."

        return text

    def _build_title(self, keywords: List[str],
                     keyword_roles: Dict[str, str]) -> str:
        """Build an evocative title from keywords."""
        nouns = [kw for kw, pos in keyword_roles.items() if pos in ("NOUN", "PROPN")]
        adjectives = [kw for kw, pos in keyword_roles.items() if pos == "ADJ"]
        verbs = [kw for kw, pos in keyword_roles.items() if pos == "VERB"]

        if adjectives and nouns:
            return f"The {adjectives[0].capitalize()} {nouns[0].capitalize()}"
        elif len(nouns) >= 2:
            return f"The {nouns[0].capitalize()} and the {nouns[1].capitalize()}"
        elif nouns:
            return f"The Tale of the {nouns[0].capitalize()}"
        elif verbs:
            return f"A Story of {verbs[0].capitalize()}ing"
        else:
            return f"The Tale of {keywords[0].capitalize()}" if keywords else "A Fairy Tale"

    def _story_to_panels(self, sentences: List[str],
                         keywords: List[str]) -> List[Dict[str, Any]]:
        """Split story into comic panels."""
        # Group sentences into panels (2-3 sentences each)
        panels = []
        group_size = max(2, len(sentences) // 6)
        
        for i in range(0, len(sentences), group_size):
            group = sentences[i:i + group_size]
            caption = " ".join(group)
            panel_num = len(panels) + 1

            # Generate image prompt
            prompt = self._panel_prompt(caption, keywords, panel_num, len(sentences) // group_size)

            panels.append({
                "id": f"panel_{panel_num}",
                "panel_number": panel_num,
                "caption": caption,
                "prompt": prompt,
                "image_url": None,
            })

        return panels

    def _panel_prompt(self, caption: str, keywords: List[str],
                      panel_num: int, total_panels: int) -> str:
        """Generate an image prompt for a story panel."""
        style = "fairy tale illustration, storybook art, warm colors, detailed fantasy"
        scene = caption[:150]
        kw_str = ", ".join(keywords[:3])
        return f"{style}, {scene}, featuring {kw_str}, panel {panel_num} of {total_panels}"

    def _fallback_story(self, keywords: List[str]) -> Dict[str, Any]:
        """Simple template fallback if model is not trained."""
        kw_str = ", ".join(keywords)
        story = (
            f"Once upon a time, in a land filled with {keywords[0] if keywords else 'wonder'}, "
            f"there lived a brave soul who sought {keywords[1] if len(keywords) > 1 else 'adventure'}. "
            f"Along the way, they encountered {keywords[2] if len(keywords) > 2 else 'many challenges'}. "
            f"With courage and wisdom, they overcame every obstacle. "
            f"And so the tale of {kw_str} came to a happy end."
        )
        return {
            "title": f"The Tale of {keywords[0].capitalize()}" if keywords else "A Tale",
            "summary": story.split('.')[0] + ".",
            "story": story,
            "panels": [{
                "id": "panel_1", "panel_number": 1,
                "caption": story, "prompt": f"fairy tale about {kw_str}",
                "image_url": None,
            }],
            "keywords_used": keywords,
            "nlp_info": {"technique": "Template fallback", "corpus": "N/A", "keyword_roles": {}},
        }
