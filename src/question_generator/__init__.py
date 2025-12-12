"""
NLP Question Generator
======================

A comprehensive question generation system using multiple NLP approaches:
- Rule-based generation using linguistic patterns
- Transformer-based generation using pre-trained models
- Hybrid approaches combining multiple techniques

This package provides tools for generating high-quality questions from text,
suitable for educational content, reading comprehension, and chatbot training.
"""

__version__ = "1.0.0"
__author__ = "AI Master's Student"

from .models.base import QuestionGenerator
from .models.rule_based import RuleBasedGenerator
from .models.transformer_based import TransformerQuestionGenerator

__all__ = [
    "QuestionGenerator",
    "RuleBasedGenerator",
    "TransformerQuestionGenerator",
]
