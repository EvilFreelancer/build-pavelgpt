import requests
import json

# from typing import List, Dict

DEFAULT_VOCAB_NAME = 'vocab.json'


class GPT2Tokenizer:
    vocab: dict
    bacov: dict

    @classmethod
    def from_pretrained(cls, model_name: str):
        config_path = f"https://huggingface.co/{model_name}/raw/main/{DEFAULT_VOCAB_NAME}"
        session = requests.Session()
        response = session.get(config_path)
        cls.vocab = json.loads(response.content)
        cls.bacov = dict((v, k) for k, v in cls.vocab.items())
        return cls

    def encode(self, text: str) -> list[int]:
        pass

    @classmethod
    def decode(cls, tokens: list[int]) -> str:
        decoded = ' '.join([cls.bacov[token] for token in tokens])
        return decoded
