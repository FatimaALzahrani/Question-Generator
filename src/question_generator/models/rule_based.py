from typing import List, Dict, Any, Optional
from .base import QuestionGenerator, Question
from ..preprocessing.text_processor import TextProcessor
from ..preprocessing.feature_extractor import FeatureExtractor


class RuleBasedGenerator(QuestionGenerator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.text_processor = TextProcessor()
        self.feature_extractor = FeatureExtractor(self.text_processor)
        self.min_sentence_length = self.config.get('min_sentence_length', 10)

    def initialize(self) -> None:
        if not self._is_initialized:
            self.text_processor.initialize()
            self._is_initialized = True

    def generate(
        self,
        text: str,
        num_questions: int = 5,
        **kwargs
    ) -> List[Question]:
        if not self._is_initialized:
            self.initialize()

        questions = []
        sentences = self.text_processor.segment_sentences(text)

        for sentence in sentences:
            if len(sentence.split()) < self.min_sentence_length:
                continue

            sentence_questions = self._generate_from_sentence(sentence, text)
            questions.extend(sentence_questions)

        questions.sort(key=lambda q: q.confidence or 0, reverse=True)
        return questions[:num_questions]

    def _generate_from_sentence(self, sentence: str, context: str) -> List[Question]:
        questions = []
        questionable_content = self.feature_extractor.extract_questionable_content(sentence)

        for item in questionable_content:
            question_text = self._create_question(sentence, item)
            if question_text:
                questions.append(Question(
                    question=question_text,
                    answer=item['content'],
                    context=context,
                    question_type=item['question_type'],
                    confidence=0.7,
                    metadata={'method': 'rule-based', 'item_type': item['type']}
                ))

        return questions

    def _create_question(self, sentence: str, item: Dict[str, Any]) -> Optional[str]:
        question_type = item['question_type']
        answer = item['content']

        question_templates = {
            'who': f"Who {self._extract_predicate(sentence, answer)}?",
            'what': f"What {self._extract_predicate(sentence, answer)}?",
            'when': f"When {self._extract_predicate(sentence, answer)}?",
            'where': f"Where {self._extract_predicate(sentence, answer)}?",
            'how many': f"How many {self._extract_predicate(sentence, answer)}?",
            'how much': f"How much {self._extract_predicate(sentence, answer)}?"
        }

        if question_type in question_templates:
            question = question_templates[question_type]
            if self._is_valid_question(question):
                return question

        return None

    def _extract_predicate(self, sentence: str, answer: str) -> str:
        doc = self.text_processor.process(sentence)

        for token in doc:
            if token.pos_ == 'VERB':
                predicate_parts = []
                for child in token.children:
                    if child.text.lower() != answer.lower():
                        predicate_parts.append(child.text)

                if predicate_parts:
                    return f"{token.text} {' '.join(predicate_parts)}"
                return token.text

        sentence_without_answer = sentence.replace(answer, "").strip()
        if sentence_without_answer:
            return sentence_without_answer.rstrip('.!?')

        return "is this"

    def _is_valid_question(self, question: str) -> bool:
        if not question or len(question) < 10:
            return False

        if not question.endswith('?'):
            return False

        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        first_word = question.lower().split()[0] if question.split() else ""

        return first_word in question_words
