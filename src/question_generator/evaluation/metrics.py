from typing import List, Dict, Any, Optional
import numpy as np
from collections import Counter
import warnings

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from sacrebleu import corpus_bleu
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

from ..models.base import Question


class QuestionEvaluator:
    def __init__(self):
        self.rouge_scorer = None
        if ROUGE_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

    def evaluate_against_reference(
        self,
        generated: List[str],
        reference: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        if metrics is None:
            metrics = ['bleu', 'rouge', 'meteor', 'bertscore']

        results = {}

        if 'bleu' in metrics and BLEU_AVAILABLE:
            bleu = self._compute_bleu(generated, reference)
            results['bleu'] = bleu

        if 'rouge' in metrics and ROUGE_AVAILABLE:
            rouge = self._compute_rouge(generated, reference)
            results.update(rouge)

        if 'meteor' in metrics:
            meteor = self._compute_meteor(generated, reference)
            results['meteor'] = meteor

        if 'bertscore' in metrics and BERTSCORE_AVAILABLE:
            bert_scores = self._compute_bertscore(generated, reference)
            results.update(bert_scores)

        return results

    def evaluate_quality(
        self,
        questions: List[Question],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        results = {
            'num_questions': len(questions),
            'diversity': self._compute_diversity([q.question for q in questions]),
            'avg_length': self._compute_avg_length([q.question for q in questions]),
            'question_type_distribution': self._compute_type_distribution(questions),
            'grammaticality': self._compute_grammaticality([q.question for q in questions]),
        }

        confidences = [q.confidence for q in questions if q.confidence is not None]
        if confidences:
            results['avg_confidence'] = np.mean(confidences)

        return results

    def _compute_bleu(self, generated: List[str], reference: List[str]) -> float:
        if not BLEU_AVAILABLE:
            return 0.0

        refs = [[ref] for ref in reference]

        min_len = min(len(generated), len(reference))
        generated = generated[:min_len]
        refs = refs[:min_len]

        if not generated:
            return 0.0

        try:
            bleu = corpus_bleu(generated, list(zip(*refs)))
            return bleu.score / 100.0
        except Exception:
            return 0.0

    def _compute_rouge(
        self,
        generated: List[str],
        reference: List[str]
    ) -> Dict[str, float]:
        if not ROUGE_AVAILABLE or self.rouge_scorer is None:
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }

        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        min_len = min(len(generated), len(reference))
        for i in range(min_len):
            score = self.rouge_scorer.score(reference[i], generated[i])
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(scores['rouge1']) if scores['rouge1'] else 0.0,
            'rouge2': np.mean(scores['rouge2']) if scores['rouge2'] else 0.0,
            'rougeL': np.mean(scores['rougeL']) if scores['rougeL'] else 0.0
        }

    def _compute_meteor(self, generated: List[str], reference: List[str]) -> float:
        scores = []
        min_len = min(len(generated), len(reference))

        for i in range(min_len):
            gen_tokens = set(generated[i].lower().split())
            ref_tokens = set(reference[i].lower().split())

            if not ref_tokens:
                continue

            matches = len(gen_tokens & ref_tokens)
            precision = matches / len(gen_tokens) if gen_tokens else 0
            recall = matches / len(ref_tokens) if ref_tokens else 0

            if precision + recall > 0:
                f_mean = (10 * precision * recall) / (9 * precision + recall)
                scores.append(f_mean)

        return np.mean(scores) if scores else 0.0

    def _compute_bertscore(
        self,
        generated: List[str],
        reference: List[str]
    ) -> Dict[str, float]:
        if not BERTSCORE_AVAILABLE:
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }

        min_len = min(len(generated), len(reference))
        if min_len == 0:
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }

        try:
            P, R, F1 = bert_score(
                generated[:min_len],
                reference[:min_len],
                lang='en',
                verbose=False
            )

            return {
                'bertscore_precision': P.mean().item(),
                'bertscore_recall': R.mean().item(),
                'bertscore_f1': F1.mean().item()
            }
        except Exception:
            return {
                'bertscore_precision': 0.0,
                'bertscore_recall': 0.0,
                'bertscore_f1': 0.0
            }

    def _compute_diversity(self, questions: List[str]) -> float:
        if not questions:
            return 0.0

        all_tokens = []
        for q in questions:
            all_tokens.extend(q.lower().split())

        if not all_tokens:
            return 0.0

        distinct_1 = len(set(all_tokens)) / len(all_tokens)

        bigrams = []
        for q in questions:
            tokens = q.lower().split()
            bigrams.extend([f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)])

        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

        return (distinct_1 + distinct_2) / 2.0

    def _compute_avg_length(self, questions: List[str]) -> float:
        if not questions:
            return 0.0

        lengths = [len(q.split()) for q in questions]
        return np.mean(lengths)

    def _compute_type_distribution(self, questions: List[Question]) -> Dict[str, int]:
        types = [q.question_type for q in questions if q.question_type]
        return dict(Counter(types))

    def _compute_grammaticality(self, questions: List[str]) -> float:
        if not questions:
            return 0.0

        scores = []
        for q in questions:
            score = 1.0

            if not q.strip():
                score = 0.0
            elif not q.endswith('?'):
                score -= 0.3
            elif not q[0].isupper():
                score -= 0.2

            first_word = q.split()[0].lower() if q.split() else ""
            question_words = ['what', 'when', 'where', 'who', 'why', 'how',
                            'which', 'whose', 'whom', 'is', 'are', 'was',
                            'were', 'do', 'does', 'did', 'can', 'could',
                            'would', 'should']

            if first_word not in question_words:
                score -= 0.3

            scores.append(max(0.0, score))

        return np.mean(scores)

    def compare_models(
        self,
        model_outputs: Dict[str, List[Question]],
        reference: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        results = {}

        for model_name, questions in model_outputs.items():
            quality = self.evaluate_quality(questions)

            if reference:
                generated_text = [q.question for q in questions]
                ref_metrics = self.evaluate_against_reference(
                    generated_text[:len(reference)],
                    reference
                )
                quality.update(ref_metrics)

            results[model_name] = quality

        return results
