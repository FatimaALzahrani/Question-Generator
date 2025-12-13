import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class FineTunedQuestionGenerator:
    def __init__(self, model_path: str = None):
        if model_path is None:
            for path in [
                "models/flan-t5-qg-fast",
                "models/flan-t5-qg-finetuned",
                "models/flan-t5-qg-test"
            ]:
                if Path(path).exists() and (Path(path) / "config.json").exists():
                    model_path = path
                    break

        if model_path is None:
            sys.exit(1)

        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)

    def generate_question(
        self,
        context: str,
        answer: str,
        num_questions: int = 1
    ) -> list:
        highlighted = context.replace(answer, f"<hl> {answer} <hl>")
        input_text = f"generate question: {highlighted}"

        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )

        outputs = self.model.generate(
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
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return questions


def demo():
    generator = FineTunedQuestionGenerator()
    generator.load_model()

    examples = [
        {
            "context": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to extract progressively higher-level features from raw input. For example, in image recognition, lower layers may identify edges, while higher layers may identify concepts relevant to humans such as digits, letters, or faces.",
            "answer": "neural networks with multiple layers"
        },
        {
            "context": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequential data and has become the foundation for modern NLP models like BERT, GPT, and T5.",
            "answer": "2017"
        },
        {
            "context": "Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, machine learning, and automation.",
            "answer": "web development, data science, machine learning, and automation"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}")
        print('='*60)

        context = example["context"]
        answer = example["answer"]

        print(f"\nContext:\n{context}")
        print(f"\nAnswer: {answer}")

        questions = generator.generate_question(context, answer, num_questions=3)

        print(f"\nGenerated Questions:")
        for j, q in enumerate(questions, 1):
            if not q.endswith('?'):
                q += '?'
            print(f"  {j}. {q}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generator = FineTunedQuestionGenerator(model_path=sys.argv[1])
    else:
        generator = FineTunedQuestionGenerator()

    generator.load_model()

    examples = [
        {
            "context": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to extract progressively higher-level features from raw input. For example, in image recognition, lower layers may identify edges, while higher layers may identify concepts relevant to humans such as digits, letters, or faces.",
            "answer": "neural networks with multiple layers"
        },
        {
            "context": "The transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequential data and has become the foundation for modern NLP models like BERT, GPT, and T5.",
            "answer": "2017"
        },
        {
            "context": "Python is a high-level, interpreted programming language known for its clear syntax and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, machine learning, and automation.",
            "answer": "web development, data science, machine learning, and automation"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}")
        print('='*60)
        print(f"\nContext:\n{example['context']}")
        print(f"\nAnswer: {example['answer']}")

        questions = generator.generate_question(example['context'], example['answer'], num_questions=3)

        print(f"\nGenerated Questions:")
        for j, q in enumerate(questions, 1):
            if not q.endswith('?'):
                q += '?'
            print(f"  {j}. {q}")
