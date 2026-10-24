import torch
import torch.nn as nn

class Committor(nn.Module):
    def __init__(self,*, num_layers, hidden_size, d, chi_A, chi_B):

        super().__init__()

        layers= [nn.Linear(d, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]

        for _ in range(num_layers-1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.Tanh()]

        layers += [nn.Linear(hidden_size, 1), nn.Sigmoid()]

        self.net= nn.Sequential(*layers)
        self.chi_A= chi_A
        self.chi_B= chi_B

    def forward(self, x):
        return (1 - self.chi_A(x)) * ((1 - self.chi_B(x)) * self.net(x).squeeze(-1) + self.chi_B(x))


# def loss_fn(model, )



def chi(s, *, c, r, w, alpha=4):
    '''
    Smoothed characteristic function of the ball B_r(c) in CV space.

    Returns s(y) = 1/2 - 1/2 tanh(y) with y = A |s - c|^2 + B, chosen
    so that y = -alpha at |s - c| = r and y = +alpha at |s - c| = r + w.
    '''
    c = torch.as_tensor(c, dtype=torch.get_default_dtype())
    A = 2 * alpha / (2 * r * w + w**2)
    B = - alpha - A * r**2

    return 0.5 - 0.5 * torch.tanh(A * torch.sum((s - c)**2, dim=-1) + B)
