import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    if config_path is None:
        possible_paths = [
            Path('config.yaml'),
            Path('config/config.yaml'),
            Path(__file__).parent.parent.parent.parent / 'config.yaml'
        ]

        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        return get_default_config()


def save_config(config: Dict[str, Any], config_path: str) -> None:
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Dict[str, Any]:
    return {
        'models': {
            'transformer': {
                'model_name': 'valhalla/t5-base-qg-hl',
                'max_length': 512,
                'num_beams': 4,
                'num_return_sequences': 3,
                'temperature': 1.0,
                'top_k': 50,
                'top_p': 0.95,
                'early_stopping': True
            },
            'rule_based': {
                'min_sentence_length': 10,
                'max_questions_per_sentence': 3,
                'question_types': ['what', 'who', 'when', 'where', 'why', 'how']
            }
        },
        'preprocessing': {
            'spacy_model': 'en_core_web_sm',
            'max_tokens': 512,
            'remove_stopwords': False,
            'lowercase': False
        },
        'evaluation': {
            'metrics': ['bleu', 'rouge', 'meteor', 'bertscore'],
            'rouge_types': ['rouge1', 'rouge2', 'rougeL']
        },
        'output': {
            'format': 'json',
            'include_confidence': True,
            'include_answer_span': True,
            'verbose': True
        }
    }
