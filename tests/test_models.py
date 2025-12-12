import pytest
from question_generator.models.rule_based import RuleBasedGenerator
from question_generator.models.transformer_based import TransformerQuestionGenerator
from question_generator.preprocessing.text_processor import TextProcessor
from question_generator.preprocessing.feature_extractor import FeatureExtractor
from question_generator.evaluation.metrics import QuestionEvaluator


SAMPLE_TEXTS = {
    'simple': "Paris is the capital of France.",
    'complex': "Albert Einstein developed the theory of relativity in 1905, which revolutionized modern physics.",
    'multi_sentence': "The Amazon rainforest is the largest tropical rainforest in the world. It spans across nine countries in South America. The forest plays a crucial role in regulating the global climate.",
}


class TestTextProcessor:

    def setup_method(self):
        self.processor = TextProcessor()
        self.processor.initialize()

    def test_sentence_segmentation(self):
        sentences = self.processor.segment_sentences(SAMPLE_TEXTS['multi_sentence'])
        assert len(sentences) == 3
        assert "Amazon rainforest" in sentences[0]

    def test_entity_extraction(self):
        entities = self.processor.extract_entities(SAMPLE_TEXTS['simple'])
        assert len(entities) > 0
        entity_texts = [e['text'] for e in entities]
        assert 'Paris' in entity_texts or 'France' in entity_texts

    def test_noun_phrase_extraction(self):
        noun_phrases = self.processor.extract_noun_phrases(SAMPLE_TEXTS['simple'])
        assert len(noun_phrases) > 0

    def test_pos_tagging(self):
        pos_tags = self.processor.get_pos_tags(SAMPLE_TEXTS['simple'])
        assert len(pos_tags) > 0
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in pos_tags)

    def test_question_detection(self):
        assert self.processor.is_question("What is the capital of France?")
        assert not self.processor.is_question("Paris is the capital.")

    def test_clean_text(self):
        dirty_text = "This   has  extra    spaces!"
        clean = self.processor.clean_text(dirty_text)
        assert "  " not in clean


class TestFeatureExtractor:

    def setup_method(self):
        self.extractor = FeatureExtractor()

    def test_extract_questionable_content(self):
        content = self.extractor.extract_questionable_content(SAMPLE_TEXTS['complex'])
        assert len(content) > 0
        assert any(item['type'] == 'entity' for item in content)

    def test_rank_sentences(self):
        ranked = self.extractor.rank_sentences(SAMPLE_TEXTS['multi_sentence'])
        assert len(ranked) == 3
        assert all('score' in item for item in ranked)
        scores = [item['score'] for item in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_identify_answer_candidates(self):
        candidates = self.extractor.identify_answer_candidates(SAMPLE_TEXTS['simple'])
        assert len(candidates) > 0
        assert all('text' in c and 'type' in c for c in candidates)


class TestRuleBasedGenerator:

    def setup_method(self):
        self.generator = RuleBasedGenerator()
        self.generator.initialize()

    def test_initialization(self):
        assert self.generator._is_initialized
        assert self.generator.text_processor is not None

    def test_question_generation(self):
        questions = self.generator.generate(SAMPLE_TEXTS['simple'], num_questions=3)
        assert len(questions) > 0
        assert all(q.question.endswith('?') for q in questions)

    def test_entity_based_questions(self):
        questions = self.generator.generate(SAMPLE_TEXTS['complex'], num_questions=5)
        assert len(questions) > 0

    def test_question_metadata(self):
        questions = self.generator.generate(SAMPLE_TEXTS['simple'], num_questions=3)
        for q in questions:
            assert q.question is not None
            assert q.context is not None
            assert q.confidence is not None

    def test_multi_sentence_processing(self):
        questions = self.generator.generate(SAMPLE_TEXTS['multi_sentence'], num_questions=5)
        assert len(questions) > 0


class TestTransformerQuestionGenerator:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        config = {
            'model_name': 'google/flan-t5-small', 
            'max_length': 256,
            'num_beams': 2,
            'num_return_sequences': 2
        }
        self.generator = TransformerQuestionGenerator(config)

    def test_initialization(self):
        self.generator.initialize()
        assert self.generator._is_initialized
        assert self.generator.model is not None
        assert self.generator.tokenizer is not None

    def test_question_generation(self):
        self.generator.initialize()
        questions = self.generator.generate(SAMPLE_TEXTS['simple'], num_questions=2)
        assert len(questions) > 0
        assert all(q.question.endswith('?') for q in questions)

    def test_model_info(self):
        info = self.generator.get_model_info()
        assert 'model_name' in info
        assert 'device' in info
        assert 'parameters' in info

    @pytest.mark.slow
    def test_batch_generation(self):
        self.generator.initialize()
        texts = [SAMPLE_TEXTS['simple'], SAMPLE_TEXTS['complex']]
        results = self.generator.batch_generate(texts, num_questions=2)
        assert len(results) == 2
        assert all(len(qs) > 0 for qs in results)


class TestQuestionEvaluator:

    def setup_method(self):
        self.evaluator = QuestionEvaluator()

    def test_quality_evaluation(self):
        from question_generator.models.base import Question

        questions = [
            Question(question="What is the capital of France?", question_type="what"),
            Question(question="When did Einstein develop relativity?", question_type="when"),
            Question(question="Where is the Amazon rainforest?", question_type="where"),
        ]

        results = self.evaluator.evaluate_quality(questions)
        assert 'num_questions' in results
        assert results['num_questions'] == 3
        assert 'diversity' in results
        assert 'question_type_distribution' in results

    def test_reference_evaluation(self):
        generated = [
            "What is the capital of France?",
            "Where is Paris located?"
        ]
        reference = [
            "What is the capital city of France?",
            "In which country is Paris?"
        ]

        results = self.evaluator.evaluate_against_reference(generated, reference)
        assert isinstance(results, dict)
        assert len(results) > 0

    def test_diversity_computation(self):
        questions = [
            "What is X?",
            "What is Y?",
            "What is Z?",
        ]
        diversity = self.evaluator._compute_diversity(questions)
        assert 0.0 <= diversity <= 1.0

        questions_diverse = [
            "What is the capital?",
            "When did this happen?",
            "How many people attended?",
        ]
        diversity_high = self.evaluator._compute_diversity(questions_diverse)
        assert diversity_high > diversity

    def test_grammaticality_check(self):
        good_questions = [
            "What is the capital of France?",
            "When did World War II end?",
        ]
        score = self.evaluator._compute_grammaticality(good_questions)
        assert score > 0.7  

        bad_questions = [
            "what the capital",  
            "france where",
        ]
        bad_score = self.evaluator._compute_grammaticality(bad_questions)
        assert bad_score < score


class TestIntegration:

    def test_end_to_end_rule_based(self):
        generator = RuleBasedGenerator()
        generator.initialize()

        questions = generator.generate(SAMPLE_TEXTS['complex'], num_questions=3)

        assert len(questions) > 0
        assert all(hasattr(q, 'question') for q in questions)
        assert all(hasattr(q, 'answer') for q in questions)
        assert all(hasattr(q, 'context') for q in questions)

    def test_end_to_end_with_evaluation(self):
        generator = RuleBasedGenerator()
        generator.initialize()

        questions = generator.generate(SAMPLE_TEXTS['simple'], num_questions=3)

        evaluator = QuestionEvaluator()
        results = evaluator.evaluate_quality(questions)

        assert 'num_questions' in results
        assert 'diversity' in results
        assert results['num_questions'] == len(questions)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
