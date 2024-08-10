import json
from dataclasses import dataclass
import requests


@dataclass
class GPT2Config:
    model_type: str = "gpt2"
    model_path: str = "gpt2"
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50264  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension

    def __init__(self, model_path: str):
        super().__init__()
        self.from_pretrained(model_path)

    def from_pretrained(cls, model_path: str) -> object:
        # TODO: HF download tools
        config_path = f"https://huggingface.co/{model_path}/raw/main/config.json"

        # Read config from remote
        session = requests.Session()
        response = session.get(config_path)

        # Convert raw JSON to object
        config = json.loads(response.content)

        cls.model_path = model_path
        if 'model_type' in config:
            cls.model_type = config['model_type']
        if 'n_ctx' in config:
            cls.block_size = config['n_ctx']
        if 'vocab_size' in config:
            cls.vocab_size = config['vocab_size']
        if 'n_layer' in config:
            cls.n_layer = config['n_layer']
        if 'n_head' in config:
            cls.n_head = config['n_head']
        if 'n_embd' in config:
            cls.n_embd = config['n_embd']

        return cls


if __name__ == "__main__":
    config = GPT2Config("gpt2")
    print(config)
