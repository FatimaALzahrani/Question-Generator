import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from question_generator.models.rule_based import RuleBasedGenerator
from question_generator.models.transformer_based import TransformerQuestionGenerator
from question_generator.models.advanced_generator import AdvancedQuestionGenerator
from question_generator.evaluation.metrics import QuestionEvaluator



def display_questions(questions, model_name):
    if not questions:
        print("  No questions generated")
        return

    for i, q in enumerate(questions, 1):
        print(f"\nQ{i}. {q.question}")
        if q.answer:
            print(f"    Answer: {q.answer}")
        print(f"    Type: {q.question_type} | Confidence: {q.confidence:.2f if q.confidence else 0:.2f}")

    print()


def main():
    text = """
    Machine learning is a subset of artificial intelligence that provides systems
    the ability to automatically learn and improve from experience without being
    explicitly programmed. Machine learning algorithms build a model based on sample
    data, known as training data, to make predictions or decisions. The field of
    machine learning was pioneered by Arthur Samuel in 1959. Today, machine learning
    is used in a wide variety of applications, including email filtering, computer
    vision, speech recognition, and recommendation systems. Deep learning, a subset
    of machine learning, has become particularly successful in recent years due to
    the availability of large datasets and powerful computing resources.
    """

    print("\nSample Text:")
    print("-" * 80)
    print(text.strip())
    print()

    num_questions = 6
    evaluator = QuestionEvaluator()

    print("\nInitializing Rule-Based Generator...")
    rule_gen = RuleBasedGenerator()
    rule_gen.initialize()
    rule_questions = rule_gen.generate(text, num_questions=num_questions)
    display_questions(rule_questions, "Rule-Based Generator")
    rule_metrics = evaluator.evaluate_quality(rule_questions)

    print("\nInitializing Standard Transformer (T5-Small)...")
    trans_config = {
        'model_name': 'google/flan-t5-small',
        'num_beams': 3,
        'num_return_sequences': 2
    }
    trans_gen = TransformerQuestionGenerator(trans_config)
    trans_gen.initialize()
    trans_questions = trans_gen.generate(text, num_questions=num_questions)
    display_questions(trans_questions, "Standard Transformer")
    trans_metrics = evaluator.evaluate_quality(trans_questions)

    print("\nInitializing Advanced Generator...")
    adv_config = {
        'model_name': 'valhalla/t5-base-qg-hl',
        'num_beams': 5,
        'num_return_sequences': 4,
        'temperature': 0.8,
        'repetition_penalty': 1.2
    }
    adv_gen = AdvancedQuestionGenerator(adv_config)
    adv_gen.initialize()
    adv_questions = adv_gen.generate(text, num_questions=num_questions, use_few_shot=True)
    display_questions(adv_questions, "Advanced Generator")
    adv_metrics = evaluator.evaluate_quality(adv_questions)

    print("\nQuality Comparison")

    metrics = ['Diversity', 'Avg Length', 'Grammaticality', 'Avg Confidence']

    print(f"\n{'Metric':<20} {'Rule-Based':<15} {'Standard':<15} {'Advanced':<15}")
    print("-" * 80)

    print(f"{'Diversity':<20} {rule_metrics['diversity']:<15.3f} {trans_metrics['diversity']:<15.3f} {adv_metrics['diversity']:<15.3f}")
    print(f"{'Avg Length':<20} {rule_metrics['avg_length']:<15.1f} {trans_metrics['avg_length']:<15.1f} {adv_metrics['avg_length']:<15.1f}")
    print(f"{'Grammaticality':<20} {rule_metrics['grammaticality']:<15.3f} {trans_metrics['grammaticality']:<15.3f} {adv_metrics['grammaticality']:<15.3f}")

    rule_conf = rule_metrics.get('avg_confidence', 0)
    trans_conf = trans_metrics.get('avg_confidence', 0)
    adv_conf = adv_metrics.get('avg_confidence', 0)
    print(f"{'Avg Confidence':<20} {rule_conf:<15.3f} {trans_conf:<15.3f} {adv_conf:<15.3f}")

    print("Question Type Distribution:")

    all_types = set()
    all_types.update(rule_metrics['question_type_distribution'].keys())
    all_types.update(trans_metrics['question_type_distribution'].keys())
    all_types.update(adv_metrics['question_type_distribution'].keys())

    print(f"\n{'Type':<15} {'Rule-Based':<15} {'Standard':<15} {'Advanced':<15}")
    print("-" * 80)
    for qtype in sorted(all_types):
        rule_count = rule_metrics['question_type_distribution'].get(qtype, 0)
        trans_count = trans_metrics['question_type_distribution'].get(qtype, 0)
        adv_count = adv_metrics['question_type_distribution'].get(qtype, 0)
        print(f"{qtype:<15} {rule_count:<15} {trans_count:<15} {adv_count:<15}")


if __name__ == "__main__":
    main()
