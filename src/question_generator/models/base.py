from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


@dataclass
class Question:
    question: str
    answer: Optional[str] = None
    context: Optional[str] = None
    question_type: Optional[str] = None
    confidence: Optional[float] = None
    answer_span: Optional[tuple] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'question': self.question,
            'answer': self.answer,
            'context': self.context,
            'question_type': self.question_type,
            'confidence': self.confidence,
            'answer_span': self.answer_span,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Question':
        return cls(
            question=data['question'],
            answer=data.get('answer'),
            context=data.get('context'),
            question_type=data.get('question_type'),
            confidence=data.get('confidence'),
            answer_span=data.get('answer_span'),
            metadata=data.get('metadata')
        )


class QuestionGenerator(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        num_questions: int = 5,
        **kwargs
    ) -> List[Question]:
        pass
