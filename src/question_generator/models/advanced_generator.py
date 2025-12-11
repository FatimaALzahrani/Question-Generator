import warnings
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    logging as transformers_logging
)
from ..models.base import QuestionGenerator, Question
from ..preprocessing.text_processor import TextProcessor

transformers_logging.set_verbosity_error()
warnings.filterwarnings('ignore')


class AdvancedQuestionGenerator(QuestionGenerator):

    FEW_SHOT_EXAMPLES = """
Example 1:
Context: Albert Einstein developed the theory of relativity in 1905.
Answer: Albert Einstein
Question: Who developed the theory of relativity?

Example 2:
Context: The Amazon rainforest spans across 9 countries and covers 5.5 million square kilometers.
Answer: 5.5 million square kilometers
Question: How large is the Amazon rainforest?

Example 3:
Context: Python was created by Guido van Rossum and first released in 1991.
Answer: 1991
Question: When was Python first released?
"""

    ADVANCED_PROMPTS = {
        'factoid': "Generate a specific factual question that tests knowledge of key information:",
        'reasoning': "Generate a question that requires reasoning or inference:",
        'analytical': "Generate an analytical question that requires deeper understanding:",
        'causal': "Generate a question about cause and effect:",
        'comparative': "Generate a comparative question:",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = config.get('model_name', 'valhalla/t5-base-qg-hl') if config else 'valhalla/t5-base-qg-hl'
        self.max_length = config.get('max_length', 512) if config else 512
        self.num_beams = config.get('num_beams', 5) if config else 5
        self.num_return_sequences = config.get('num_return_sequences', 4) if config else 4
        self.temperature = config.get('temperature', 0.8) if config else 0.8
        self.top_k = config.get('top_k', 50) if config else 50
        self.top_p = config.get('top_p', 0.92) if config else 0.92
        self.repetition_penalty = config.get('repetition_penalty', 1.2) if config else 1.2
        self.length_penalty = config.get('length_penalty', 1.0) if config else 1.0

        self.min_question_quality = 0.6
        self.min_answer_confidence = 0.5

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.text_processor = TextProcessor()

    def initialize(self) -> None:
        if not self._is_initialized:
            print(f"Loading {self.model_name}")

            models_to_try = [
                self.model_name,
                "google/flan-t5-base",
                "t5-base"
            ]

            for model_name in models_to_try:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                    self.model.to(self.device)
                    self.model.eval()
                    self.text_processor.initialize()
                    self._is_initialized = True
                    self.model_name = model_name
                    print(f"Loaded {model_name}\n")
                    return
                except Exception as e:
                    if model_name == models_to_try[-1]:
                        raise RuntimeError(f"Could not load any model: {e}")
                    continue

    def generate(
        self,
        text: str,
        num_questions: int = 5,
        question_types: Optional[List[str]] = None,
        use_few_shot: bool = True,
        **kwargs
    ) -> List[Question]:
        if not self._is_initialized:
            self.initialize()

        answer_candidates = self._extract_answer_candidates(text)
        all_questions = []

        for candidate in answer_candidates[:num_questions * 2]:
            questions = self._generate_with_answer(
                text,
                candidate['text'],
                use_few_shot=use_few_shot
            )
            all_questions.extend(questions)

        if len(all_questions) < num_questions * 3:
            cot_questions = self._generate_with_chain_of_thought(text)
            all_questions.extend(cot_questions)

        filtered_questions = self._filter_by_quality(all_questions, text)
        diverse_questions = self._ensure_diversity(filtered_questions, num_questions)

        return diverse_questions[:num_questions]

    def _is_valid_answer_candidate(self, answer_text: str) -> bool:
        invalid_answers = {
            'a', 'an', 'the', 'one', 'two', 'some', 'subset', 'a subset',
            'this', 'that', 'these', 'those', 'it', 'they', 'them',
            'he', 'she', 'him', 'her', 'his', 'hers', 'its', 'their',
            'i', 'you', 'we', 'us', 'me', 'my', 'your', 'our',
            'thing', 'things', 'something', 'anything', 'nothing',
            'part', 'parts', 'type', 'types', 'kind', 'kinds'
        }

        text = answer_text.strip().lower()

        if len(text) < 2:
            return False

        if text in invalid_answers:
            return False

        words = text.split()
        if all(w in {'a', 'an', 'the', 'this', 'that', 'these', 'those'} for w in words):
            return False

        if len(words) == 1 and (len(text) == 1 or text.isdigit()):
            if not text[0].isupper():
                return False

        if not any(c.isalnum() for c in text):
            return False

        return True

    def _extract_answer_candidates(self, text: str) -> List[Dict[str, Any]]:
        doc = self.text_processor.process(text)
        candidates = []

        for ent in doc.ents:
            if not self._is_valid_answer_candidate(ent.text):
                continue

            importance = self._score_entity_importance(ent, doc)

            if importance > 0.3:
                candidates.append({
                    'text': ent.text,
                    'type': 'entity',
                    'label': ent.label_,
                    'importance': importance,
                    'context': ent.sent.text,
                    'position': ent.start_char
                })

        for chunk in doc.noun_chunks:
            if not self._is_valid_answer_candidate(chunk.text):
                continue

            if len(chunk.text.split()) > 1 and not chunk.root.pos_ == 'PRON':
                importance = self._score_phrase_importance(chunk, doc)

                if importance > 0.4:
                    candidates.append({
                        'text': chunk.text,
                        'type': 'noun_phrase',
                        'label': 'NP',
                        'importance': importance,
                        'context': chunk.sent.text,
                        'position': chunk.start_char
                    })

        candidates.sort(key=lambda x: x['importance'], reverse=True)

        seen = set()
        unique_candidates = []
        for c in candidates:
            text_lower = c['text'].lower()
            if text_lower not in seen:
                seen.add(text_lower)
                unique_candidates.append(c)

        return unique_candidates[:15]

    def _score_entity_importance(self, ent, doc) -> float:
        score = 0.5

        important_types = {'PERSON': 0.3, 'ORG': 0.2, 'GPE': 0.2, 'DATE': 0.2,
                          'MONEY': 0.3, 'CARDINAL': 0.1, 'QUANTITY': 0.2}
        score += important_types.get(ent.label_, 0)

        if ent.start == ent.sent.start:
            score += 0.1

        if len(ent.text.split()) > 1:
            score += 0.1

        return min(score, 1.0)

    def _score_phrase_importance(self, chunk, doc) -> float:
        score = 0.4

        word_count = len(chunk.text.split())
        if word_count >= 2:
            score += 0.1
        if word_count >= 3:
            score += 0.1

        has_entity = any(token.ent_type_ for token in chunk)
        if has_entity:
            score += 0.2

        if chunk.root.dep_ in ['nsubj', 'nsubjpass']:
            score += 0.1

        return min(score, 1.0)

    def _generate_with_answer(
        self,
        context: str,
        answer: str,
        use_few_shot: bool = True
    ) -> List[Question]:
        sentences = self.text_processor.segment_sentences(context)
        target_sentence = None
        for sent in sentences:
            if answer.lower() in sent.lower():
                target_sentence = sent
                break

        if not target_sentence:
            target_sentence = context[:200]

        highlighted = target_sentence.replace(answer, f"<hl> {answer} <hl>", 1)

        if use_few_shot and 'flan' in self.model_name.lower():
            input_text = f"{self.FEW_SHOT_EXAMPLES}\n\nNow generate a question:\nContext: {highlighted}\nAnswer: {answer}\nQuestion:"
        else:
            input_text = f"generate question: {highlighted}"

        questions_text = self._generate_text(input_text)

        questions = []
        for q_text in questions_text:
            if self._is_valid_question(q_text):
                questions.append(Question(
                    question=q_text,
                    answer=answer,
                    context=context,
                    question_type=self._classify_question_type(q_text),
                    confidence=0.9,
                    answer_span=self.text_processor.find_answer_span(context, answer),
                    metadata={
                        'method': 'advanced-answer-extraction',
                        'model': self.model_name,
                        'few_shot': use_few_shot
                    }
                ))

        return questions

    def _generate_with_chain_of_thought(self, text: str) -> List[Question]:
        if 'flan' in self.model_name.lower():
            prompt = f"""Let's think step by step to generate good questions from this text:

Text: {text[:300]}

Step 1: What are the key facts?
Step 2: What would someone want to know?
Step 3: Generate a clear question.

Question:"""

            questions_text = self._generate_text(prompt)

            questions = []
            for q_text in questions_text:
                if self._is_valid_question(q_text):
                    questions.append(Question(
                        question=q_text,
                        answer=None,
                        context=text,
                        question_type=self._classify_question_type(q_text),
                        confidence=0.75,
                        metadata={
                            'method': 'chain-of-thought',
                            'model': self.model_name
                        }
                    ))

            return questions

        return []

    def _generate_text(self, input_text: str) -> List[str]:
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        generation_temp = min(0.7, self.temperature)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=96,
                min_length=8,
                num_beams=self.num_beams,
                num_return_sequences=self.num_return_sequences,
                temperature=generation_temp,
                top_k=self.top_k,
                top_p=0.90,
                repetition_penalty=1.3,
                length_penalty=self.length_penalty,
                early_stopping=True,
                do_sample=True,
                no_repeat_ngram_size=3
            )

        generated = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        cleaned = []
        for text in generated:
            text = text.strip()

            if text.lower().startswith('question:'):
                text = text[9:].strip()

            if not text.endswith('?'):
                text += '?'

            if len(text) > 10 and len(text.split()) >= 3:
                cleaned.append(text)

        return cleaned

    def _is_valid_question(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False

        text_lower = text.lower()

        question_starters = ['what', 'when', 'where', 'who', 'why', 'how', 'which',
                           'whose', 'whom', 'is', 'are', 'was', 'were', 'do', 'does',
                           'did', 'can', 'could', 'would', 'should', 'will']

        starts_correctly = any(text_lower.startswith(word) for word in question_starters)
        ends_correctly = text.endswith('?')

        word_count = len(text.split())
        reasonable_length = 3 <= word_count <= 20

        has_verb = any(word in text_lower for word in ['is', 'are', 'was', 'were', 'did', 'do', 'does', 'can', 'has', 'have'])

        return starts_correctly and ends_correctly and reasonable_length and has_verb

    def _classify_question_type(self, question: str) -> str:
        q_lower = question.lower()

        if q_lower.startswith('why') or 'reason' in q_lower or 'because' in q_lower:
            return 'causal'
        elif q_lower.startswith('how') and any(word in q_lower for word in ['many', 'much', 'long', 'often']):
            return 'quantitative'
        elif q_lower.startswith('how'):
            return 'process'
        elif q_lower.startswith('what') and 'difference' in q_lower:
            return 'comparative'
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

    def _validate_answer_span(self, question: Question, context: str) -> bool:
        if not question.answer:
            return False

        answer = question.answer.strip().lower()
        question_text = question.question.strip().lower()
        context_lower = context.lower()

        if answer not in context_lower:
            return False

        q_words = set(question_text.replace('?', '').split())
        a_words = set(answer.split())

        q_words -= {'what', 'when', 'where', 'who', 'why', 'how', 'which', 'is', 'are', 'was', 'were', 'do', 'does', 'did'}

        if a_words and q_words and len(a_words - q_words) == 0:
            return False

        overlap = len(a_words & q_words) / len(a_words) if a_words else 0
        if overlap > 0.7:
            return False

        return True

    def _filter_by_quality(self, questions: List[Question], context: str) -> List[Question]:
        filtered = []

        for q in questions:
            if not self._validate_answer_span(q, context):
                continue

            score = self._compute_question_quality(q, context)

            if score >= self.min_question_quality:
                q.confidence = score
                filtered.append(q)

        return filtered

    def _compute_question_quality(self, question: Question, context: str) -> float:
        score = 0.5

        if question.answer and not self._is_valid_answer_candidate(question.answer):
            return 0.0

        word_count = len(question.question.split())
        if 5 <= word_count <= 15:
            score += 0.15
        elif 3 <= word_count <= 20:
            score += 0.05
        else:
            score -= 0.1

        if question.answer:
            score += 0.15

            if question.answer.lower() in context.lower():
                score += 0.1
            else:
                score -= 0.2

            q_words = set(question.question.lower().split())
            a_words = set(question.answer.lower().split())
            if len(a_words - q_words) < 1:
                score -= 0.15

        if question.question_type in ['causal', 'comparative', 'process']:
            score += 0.1

        q_lower = question.question.lower()
        if question.answer:
            a_lower = question.answer.lower()
            if f"what is {a_lower}" in q_lower or f"what are {a_lower}" in q_lower:
                if a_lower in ['deep learning', 'machine learning', 'neural networks']:
                    score -= 0.1

        return max(0.0, min(score, 1.0))

    def _ensure_diversity(self, questions: List[Question], target_count: int) -> List[Question]:
        by_type = {}
        for q in questions:
            qtype = q.question_type or 'other'
            if qtype not in by_type:
                by_type[qtype] = []
            by_type[qtype].append(q)

        for qtype in by_type:
            by_type[qtype].sort(key=lambda x: x.confidence or 0, reverse=True)

        diverse = []
        types_used = set()

        for qtype, qs in sorted(by_type.items(), key=lambda x: len(x[1]), reverse=True):
            if len(diverse) < target_count and qs:
                best_q = qs[0]
                if not any(self._questions_too_similar(best_q, existing) for existing in diverse):
                    diverse.append(best_q)
                    types_used.add(qtype)

        remaining = [q for q in questions if q not in diverse]
        remaining.sort(key=lambda x: x.confidence or 0, reverse=True)

        for q in remaining:
            if len(diverse) >= target_count:
                break

            is_too_similar = False
            for existing in diverse:
                if self._questions_too_similar(q, existing):
                    is_too_similar = True
                    break

            if not is_too_similar:
                if self._adds_new_information(q, diverse):
                    diverse.append(q)

        return diverse

    def _adds_new_information(self, question: Question, existing_questions: List[Question]) -> bool:
        if not question.answer:
            return True

        q_answer = question.answer.lower().strip()

        for existing in existing_questions:
            if existing.answer:
                e_answer = existing.answer.lower().strip()

                if q_answer == e_answer:
                    return False

                q_words = set(q_answer.split())
                e_words = set(e_answer.split())
                if q_words and e_words:
                    overlap = len(q_words & e_words) / max(len(q_words), len(e_words))
                    if overlap > 0.8:
                        return False

        return True

    def _questions_too_similar(self, q1: Question, q2: Question) -> bool:
        text1 = q1.question.lower().strip()
        text2 = q2.question.lower().strip()

        if text1 == text2:
            return True

        words1 = set(text1.split())
        words2 = set(text2.split())

        stop_words = {
            'what', 'when', 'where', 'who', 'why', 'how', 'which',
            'is', 'are', 'was', 'were', 'do', 'does', 'did',
            'can', 'could', 'would', 'should', 'will', 'shall',
            'a', 'an', 'the', 'of', 'to', 'in', 'for', 'on', 'at',
            '?', '.', ',', '!', ';', ':'
        }
        words1 -= stop_words
        words2 -= stop_words

        if not words1 or not words2:
            return False

        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard = intersection / union if union > 0 else 0

        same_answer = False
        if q1.answer and q2.answer:
            same_answer = q1.answer.lower().strip() == q2.answer.lower().strip()

        threshold = 0.5 if same_answer else 0.7

        return jaccard > threshold
