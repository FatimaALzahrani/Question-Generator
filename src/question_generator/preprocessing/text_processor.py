import re
from typing import List, Dict, Any, Tuple, Optional
import spacy
from spacy.tokens import Doc
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


class TextProcessor:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self._ensure_nltk_data()

    def _ensure_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

    def initialize(self):
        if self.nlp is None:
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                import subprocess
                subprocess.run(
                    ["python", "-m", "spacy", "download", self.model_name],
                    check=True, capture_output=True
                )
                self.nlp = spacy.load(self.model_name)

    def process(self, text: str) -> Doc:
        if self.nlp is None:
            self.initialize()
        return self.nlp(text)

    def segment_sentences(self, text: str) -> List[str]:
        if self.nlp is None:
            self.initialize()
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        doc = self.process(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "description": spacy.explain(ent.label_)
            })
        return entities

    def extract_noun_phrases(self, text: str) -> List[str]:
        doc = self.process(text)
        return [chunk.text for chunk in doc.noun_chunks]

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        doc = self.process(text)
        return [(token.text, token.pos_) for token in doc]

    def get_dependencies(self, text: str) -> List[Dict[str, Any]]:
        doc = self.process(text)
        dependencies = []
        for token in doc:
            dependencies.append({
                "text": token.text,
                "pos": token.pos_,
                "dep": token.dep_,
                "head": token.head.text,
                "children": [child.text for child in token.children]
            })
        return dependencies

    def find_answer_span(self, context: str, answer: str) -> Optional[Tuple[int, int]]:
        start = context.lower().find(answer.lower())
        if start != -1:
            return (start, start + len(answer))

        doc = self.process(context)
        answer_doc = self.process(answer)
        answer_tokens = [token.text.lower() for token in answer_doc]

        for i, token in enumerate(doc):
            if token.text.lower() == answer_tokens[0]:
                match = True
                for j, ans_token in enumerate(answer_tokens):
                    if i + j >= len(doc) or doc[i + j].text.lower() != ans_token:
                        match = False
                        break
                if match:
                    return (doc[i].idx, doc[i + len(answer_tokens) - 1].idx +
                            len(doc[i + len(answer_tokens) - 1].text))

        return None

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def is_question(self, text: str) -> bool:
        text = text.strip()
        if text.endswith('?'):
            return True

        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which',
                         'whose', 'whom', 'does', 'do', 'did', 'is', 'are', 'was', 'were']

        first_word = text.split()[0].lower() if text.split() else ""
        return first_word in question_words
