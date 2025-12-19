from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .base import QuestionGenerator, Question
from ..preprocessing.text_processor import TextProcessor


class TransformerQuestionGenerator(QuestionGenerator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = self.config.get('model_name', 'google/flan-t5-base')
        self.max_length = self.config.get('max_length', 512)
        self.num_beams = self.config.get('num_beams', 4)
        self.num_return_sequences = self.config.get('num_return_sequences', 3)
        self.temperature = self.config.get('temperature', 1.0)
        self.top_k = self.config.get('top_k', 50)
        self.top_p = self.config.get('top_p', 0.95)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.text_processor = TextProcessor()

    def initialize(self) -> None:
        if not self._is_initialized:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
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
        entities = self.text_processor.extract_entities(text)

        for entity in entities[:num_questions * 2]:
            answer = entity['text']
            generated_questions = self.generate_with_answer(text, answer, num_questions=2)
            questions.extend(generated_questions)

        if len(questions) < num_questions:
            context_questions = self._generate_from_context(text, num_questions - len(questions))
            questions.extend(context_questions)

        questions.sort(key=lambda q: q.confidence or 0, reverse=True)
        return questions[:num_questions]

    def generate_with_answer(
        self,
        context: str,
        answer: str,
        num_questions: int = 1
    ) -> List[Question]:
        if not self._is_initialized:
            self.initialize()

        highlighted = context.replace(answer, f"<hl> {answer} <hl>", 1)
        input_text = f"generate question: {highlighted}"

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=96,
                num_beams=self.num_beams,
                num_return_sequences=min(num_questions, self.num_return_sequences),
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                early_stopping=True,
                do_sample=True
            )

        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        questions = []
        for q_text in generated_texts:
            q_text = q_text.strip()
            if not q_text.endswith('?'):
                q_text += '?'

            if self._is_valid_question(q_text):
                questions.append(Question(
                    question=q_text,
                    answer=answer,
                    context=context,
                    question_type=self._classify_question_type(q_text),
                    confidence=0.85,
                    answer_span=self.text_processor.find_answer_span(context, answer),
                    metadata={'method': 'transformer', 'model': self.model_name}
                ))

        return questions

    def _generate_from_context(self, text: str, num_questions: int) -> List[Question]:
        input_text = f"generate questions: {text[:400]}"

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=96,
                num_beams=self.num_beams,
                num_return_sequences=min(num_questions, self.num_return_sequences),
                temperature=self.temperature,
                do_sample=True,
                early_stopping=True
            )

        generated_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        questions = []
        for q_text in generated_texts:
            q_text = q_text.strip()
            if not q_text.endswith('?'):
                q_text += '?'

            if self._is_valid_question(q_text):
                questions.append(Question(
                    question=q_text,
                    answer=None,
                    context=text,
                    question_type=self._classify_question_type(q_text),
                    confidence=0.75,
                    metadata={'method': 'transformer-context', 'model': self.model_name}
                ))

        return questions

    def _is_valid_question(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False

        if not text.endswith('?'):
            return False

        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which',
                         'whose', 'whom', 'is', 'are', 'was', 'were', 'do', 'does', 'did']

        first_word = text.lower().split()[0] if text.split() else ""
        return first_word in question_words

    def _classify_question_type(self, question: str) -> str:
        q_lower = question.lower()

        if q_lower.startswith('why'):
            return 'causal'
        elif q_lower.startswith('how'):
            return 'process'
        elif q_lower.startswith('what'):
            return 'definitional'
        elif q_lower.startswith('who'):
            return 'person'
        elif q_lower.startswith('when'):
            return 'temporal'
        elif q_lower.startswith('where'):
            return 'spatial'
        elif q_lower.startswith('which'):
            return 'selection'
        else:
            return 'yes/no'
