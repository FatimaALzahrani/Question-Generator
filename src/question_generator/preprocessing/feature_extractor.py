from typing import List, Dict, Any, Optional
from .text_processor import TextProcessor


class FeatureExtractor:
    def __init__(self, text_processor: Optional[TextProcessor] = None):
        self.processor = text_processor or TextProcessor()
        if self.processor.nlp is None:
            self.processor.initialize()

    def extract_questionable_content(self, text: str) -> List[Dict[str, Any]]:
        doc = self.processor.process(text)
        questionable_items = []

        entities = self.processor.extract_entities(text)
        for ent in entities:
            question_type = self._entity_to_question_type(ent['label'])
            questionable_items.append({
                'content': ent['text'],
                'type': 'entity',
                'entity_label': ent['label'],
                'question_type': question_type,
                'span': (ent['start'], ent['end']),
                'context': text
            })

        for sent in doc.sents:
            verb_phrases = self._extract_verb_phrases(sent)
            for vp in verb_phrases:
                questionable_items.append({
                    'content': vp['text'],
                    'type': 'verb_phrase',
                    'question_type': 'what',
                    'context': sent.text,
                    'root_verb': vp['root']
                })

        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                questionable_items.append({
                    'content': chunk.text,
                    'type': 'noun_phrase',
                    'question_type': 'what',
                    'context': chunk.sent.text
                })

        return questionable_items

    def _entity_to_question_type(self, entity_label: str) -> str:
        mapping = {
            'PERSON': 'who',
            'ORG': 'what',
            'GPE': 'where',
            'LOC': 'where',
            'DATE': 'when',
            'TIME': 'when',
            'MONEY': 'how much',
            'QUANTITY': 'how many',
            'CARDINAL': 'how many',
        }
        return mapping.get(entity_label, 'what')

    def _extract_verb_phrases(self, sent) -> List[Dict[str, str]]:
        verb_phrases = []
        for token in sent:
            if token.pos_ == 'VERB':
                phrase_tokens = [token]
                for child in token.children:
                    if child.dep_ in ['aux', 'auxpass', 'neg', 'prt']:
                        phrase_tokens.append(child)

                phrase_tokens.sort(key=lambda t: t.i)
                phrase_text = ' '.join([t.text for t in phrase_tokens])

                verb_phrases.append({
                    'text': phrase_text,
                    'root': token.lemma_
                })

        return verb_phrases
