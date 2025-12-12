import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from datasets import load_dataset
from tqdm import tqdm


def download_squad():
    print("Downloading SQuAD v1.1 dataset...")
    dataset = load_dataset("squad")
    return dataset


def convert_to_qg_format(example: Dict) -> Dict:
    context = example['context']
    question = example['question']
    answer_text = example['answers']['text'][0]
    answer_start = example['answers']['answer_start'][0]

    answer_end = answer_start + len(answer_text)
    highlighted_context = (
        context[:answer_start] +
        f"<hl> {answer_text} <hl>" +
        context[answer_end:]
    )

    return {
        'input_text': f"generate question: {highlighted_context}",
        'target_text': question,
        'context': context,
        'answer': answer_text
    }


def filter_quality(examples: List[Dict]) -> List[Dict]:
    filtered = []

    for ex in tqdm(examples, desc="Filtering"):
        context_words = len(ex['context'].split())
        question_words = len(ex['target_text'].split())
        answer_words = len(ex['answer'].split())

        if not (50 <= context_words <= 500):
            continue
        if not (5 <= question_words <= 30):
            continue
        if not (1 <= answer_words <= 50):
            continue

        if ex['answer'].lower() not in ex['context'].lower():
            continue

        if not ex['target_text'].endswith('?'):
            continue

        filtered.append(ex)

    return filtered


def prepare_squad_for_training(
    output_dir: str = "data/squad_qg",
    train_size: int = 50000,
    val_size: int = 5000,
    test_size: int = 5000
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = download_squad()

    print("\nConverting training data...")
    train_examples = []
    for example in tqdm(dataset['train']):
        converted = convert_to_qg_format(example)
        train_examples.append(converted)

    print("\nFiltering training data...")
    train_examples = filter_quality(train_examples)

    random.seed(42)
    random.shuffle(train_examples)

    train_data = train_examples[:train_size]
    val_data = train_examples[train_size:train_size + val_size]
    test_data = train_examples[train_size + val_size:train_size + val_size + test_size]

    print("\nSaving data...")

    with open(output_path / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(output_path / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    with open(output_path / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    stats = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'total_filtered': len(train_examples),
        'avg_context_length': sum(len(ex['context'].split()) for ex in train_data) / len(train_data),
        'avg_question_length': sum(len(ex['target_text'].split()) for ex in train_data) / len(train_data),
        'avg_answer_length': sum(len(ex['answer'].split()) for ex in train_data) / len(train_data)
    }

    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nData preparation complete!")
    print(f"Train: {stats['train_size']} examples")
    print(f"Val: {stats['val_size']} examples")
    print(f"Test: {stats['test_size']} examples")
    print(f"Avg context length: {stats['avg_context_length']:.1f} words")
    print(f"Avg question length: {stats['avg_question_length']:.1f} words")
    print(f"Avg answer length: {stats['avg_answer_length']:.1f} words")
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    prepare_squad_for_training(
        output_dir="data/squad_qg",
        train_size=50000,
        val_size=5000,
        test_size=5000
    )
