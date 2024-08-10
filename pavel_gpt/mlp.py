import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP)

    https://arxiv.org/abs/2404.19756
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


if __name__ == '__main__':
    class Config:
        n_embd = 128


    config = Config()
    model = MLP(config)
    input_data = torch.randn(5, config.n_embd)
    output = model(input_data)
    print(output)
