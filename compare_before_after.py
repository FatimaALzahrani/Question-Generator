import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ModelComparator:
    def __init__(
        self,
        base_model: str = "google/flan-t5-base",
        finetuned_model: str = None
    ):
        self.base_model_name = base_model

        if finetuned_model is None:
            for path in [
                "models/flan-t5-qg-fast",
                "models/flan-t5-qg-finetuned",
                "models/flan-t5-qg-test"
            ]:
                if Path(path).exists() and (Path(path) / "config.json").exists():
                    finetuned_model = path
                    break

        if finetuned_model is None:
            sys.exit(1)

        self.finetuned_model_name = finetuned_model
        self.base_tokenizer = None
        self.base_model = None
        self.finetuned_tokenizer = None
        self.finetuned_model = None

    def load_models(self):
        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_name)
        self.finetuned_tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_name)
        self.finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(self.finetuned_model_name)

    def generate_with_model(
        self,
        model,
        tokenizer,
        context: str,
        answer: str,
        num_questions: int = 3
    ):
        highlighted = context.replace(answer, f"<hl> {answer} <hl>")
        input_text = f"generate question: {highlighted}"

        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            max_length=128,
            num_return_sequences=num_questions,
            num_beams=5,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            early_stopping=True
        )

        questions = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return questions

    def compare(self, context: str, answer: str):
        print(f"Context:\n{context}")
        print(f"\nTarget Answer: {answer}")
        print("\n" + "-"*60)

        print("\nBASE MODEL:")
        base_questions = self.generate_with_model(
            self.base_model,
            self.base_tokenizer,
            context,
            answer,
            num_questions=3
        )

        for i, q in enumerate(base_questions, 1):
            if not q.endswith('?'):
                q += '?'
            print(f"  {i}. {q}")

        print("\n" + "-"*60)

        print("\nFINE-TUNED MODEL:")
        finetuned_questions = self.generate_with_model(
            self.finetuned_model,
            self.finetuned_tokenizer,
            context,
            answer,
            num_questions=3
        )

        for i, q in enumerate(finetuned_questions, 1):
            if not q.endswith('?'):
                q += '?'
            print(f"  {i}. {q}")


def main():
    comparator = ModelComparator()
    comparator.load_models()

    test_cases = [
        {
            "context": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to extract progressively higher-level features from raw input. For example, in image recognition, lower layers may identify edges, while higher layers may identify concepts relevant to humans such as digits, letters, or faces.",
            "answer": "neural networks with multiple layers"
        },
        {
            "context": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequential data and has become the foundation for modern NLP models like BERT, GPT, and T5.",
            "answer": "self-attention mechanisms"
        },
        {
            "context": "Transfer learning is a machine learning technique where a model trained on one task is adapted for a second related task. This approach is particularly effective in NLP, where models pretrained on large text corpora can be fine-tuned for specific tasks with relatively small datasets.",
            "answer": "a model trained on one task is adapted for a second related task"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}")
        print('='*60)
        comparator.compare(
            test_case["context"],
            test_case["answer"]
        )


if __name__ == "__main__":
    main()
