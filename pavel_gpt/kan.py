import torch
import torch.nn as nn


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN)
    """

    def __init__(self, config):
        super(KAN, self).__init__()
        self.knots1 = nn.Parameter(torch.linspace(-1, 1, 10))
        self.coeffs1 = nn.Parameter(torch.randn(config.n_embd, config.n_embd, 10))

        self.knots2 = nn.Parameter(torch.linspace(-1, 1, 10))
        self.coeffs2 = nn.Parameter(torch.randn(config.n_embd, config.n_embd, 10))

        self.gelu = nn.GELU(approximate='tanh')

    def spline_function(self, x, knots, coeffs):
        x_expanded = x.unsqueeze(-1).expand(-1, -1, knots.size(0))
        basis = torch.abs(x_expanded - knots)
        output = torch.einsum('bik,ijk->bij', basis, coeffs)
        return output.mean(dim=-1)

    def forward(self, x):
        x = self.spline_function(x, self.knots1, self.coeffs1)
        x = self.gelu(x)  # Применение нелинейности, как в MLP
        x = self.spline_function(x, self.knots2, self.coeffs2)
        return x


if __name__ == '__main__':
    class Config:
        n_embd = 128


    config = Config()
    model = KAN(config)
    input_data = torch.randn(5, config.n_embd)
    output = model(input_data)
    print(output)
