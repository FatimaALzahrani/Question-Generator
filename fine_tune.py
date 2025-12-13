import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import evaluate


@dataclass
class FineTuneConfig:
    model_name: str = "google/flan-t5-base"
    data_dir: str = "data/squad_qg"
    output_dir: str = "models/flan-t5-qg-finetuned"

    max_input_length: int = 512
    max_target_length: int = 128

    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-4
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01

    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100

    fp16: bool = torch.cuda.is_available()

    seed: int = 42


class QuestionGenerationTrainer:
    def __init__(self, config: FineTuneConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None

    def load_model(self):
        print(f"Loading model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)

        print(f"Model loaded: {self.model.num_parameters():,} parameters")

    def load_data(self):
        data_path = Path(self.config.data_dir)

        print("\nLoading training data...")
        with open(data_path / "train.json", "r") as f:
            train_data = json.load(f)

        print("Loading validation data...")
        with open(data_path / "val.json", "r") as f:
            val_data = json.load(f)

        print(f"Train examples: {len(train_data)}")
        print(f"Val examples: {len(val_data)}")

        self.train_dataset = Dataset.from_list(train_data)
        self.val_dataset = Dataset.from_list(val_data)

    def preprocess_function(self, examples):
        inputs = examples["input_text"]
        targets = examples["target_text"]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
            padding="max_length"
        )

        labels = self.tokenizer(
            targets,
            max_length=self.config.max_target_length,
            truncation=True,
            padding="max_length"
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def prepare_datasets(self):
        print("\nTokenizing datasets...")

        self.train_dataset = self.train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.train_dataset.column_names,
            desc="Tokenizing train"
        )

        self.val_dataset = self.val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.val_dataset.column_names,
            desc="Tokenizing validation"
        )

        print("Datasets ready!")

    def compute_metrics(self, eval_pred):
        try:
            rouge = evaluate.load("rouge")
        except:
            print("Warning: ROUGE metric not available")
            return {
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0
            }

        predictions, labels = eval_pred

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True
        )

        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True
        )

        result = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        result = {key: value * 100 for key, value in result.items()}

        return {
            "rouge1": round(result["rouge1"], 2),
            "rouge2": round(result["rouge2"], 2),
            "rougeL": round(result["rougeL"], 2)
        }

    def train(self):
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,

            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,

            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,

            load_best_model_at_end=True,
            metric_for_best_model="rouge1",
            greater_is_better=True,

            fp16=self.config.fp16,

            predict_with_generate=True,
            generation_max_length=self.config.max_target_length,

            save_total_limit=3,

            seed=self.config.seed,

            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )

        print("\nStarting training...")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"FP16: {self.config.fp16}")

        trainer.train()

        print("\nTraining complete!")

        print("\nEvaluating on validation set...")
        metrics = trainer.evaluate()

        print("\nValidation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")

        print(f"\nSaving model to: {self.config.output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)

        with open(Path(self.config.output_dir) / "training_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\nModel saved successfully!")

        return trainer


def main():
    config = FineTuneConfig()

    print("="*60)
    print("Question Generation Fine-Tuning")
    print("="*60)

    print("\nConfiguration:")
    print(f"  Model: {config.model_name}")
    print(f"  Data: {config.data_dir}")
    print(f"  Output: {config.output_dir}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")

    trainer = QuestionGenerationTrainer(config)

    trainer.load_model()
    trainer.load_data()
    trainer.prepare_datasets()

    trainer.train()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFine-tuned model saved to: {config.output_dir}")
    print("\nNext steps:")
    print("1. Test the model: python test_finetuned.py")
    print("2. Compare with base: python compare_models.py")
    print("3. Use in generator: Update model_name in config")


if __name__ == "__main__":
    main()
