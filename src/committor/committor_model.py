import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

class Committor(nn.Module):
    def __init__(self,*, num_layers, hidden_size, d, chi_A, chi_B, **kwargs):

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


def loss_fn(*, model, X, weights, create_graph=True):
    '''
    Monte Carlo estimate of the Dirichlet energy: mean of weights * |grad q|^2 over
    the points X (N, d). weights (N,) are precomputed, e.g. exp(-beta V(X)) for
    uniform samples, since they don't depend on the model.

    create_graph=False is enough when the loss is only evaluated (test set), not
    differentiated with respect to the weights of the network.
    '''
    X= X.clone().requires_grad_(True)

    q= model(X)

    # [0] since torch.autograd.grad() returns other stuff we don't need
    grad_q= torch.autograd.grad(q.sum(), X, create_graph=create_graph)[0]

    return torch.mean(weights * torch.sum(grad_q**2, dim=-1))




def chi(s, *, c, r, w, alpha=4):
    '''
    Smoothed characteristic function of the ball B_r(c) in CV space.

    Returns s(y) = 1/2 - 1/2 tanh(y) with y = A |s - c|^2 + B, chosen
    so that y = -alpha at |s - c| = r and y = +alpha at |s - c| = r + w.
    '''
    c = torch.as_tensor(c, dtype=s.dtype)
    A = 2 * alpha / (2 * r * w + w**2)
    B = - alpha - A * r**2

    return 0.5 - 0.5 * torch.tanh(A * torch.sum((s - c)**2, dim=-1) + B)


################################################################################
#                     Building, training and loading models                    #
################################################################################

def build_committor(model_params, *, indicators):
    '''
    The one place that turns saved settings into a model, used both for training
    and for loading, so the two can't disagree.

    model_params holds plain numbers only (it is saved in checkpoints): the
    Committor arguments, plus r_chi and w_chi for the smoothed indicators.
    indicators(*, r, w) is supplied by the problem module and returns the torch
    functions (chi_A, chi_B), each mapping (..., d) -> (...).
    '''
    chi_A, chi_B= indicators(r=model_params['r_chi'], w=model_params['w_chi'])
    return Committor(**model_params, chi_A=chi_A, chi_B=chi_B)


def load_committor(ckpt_file, *, indicators, problem=None):
    '''
    Rebuild the model saved in the checkpoint ckpt_file, with its trained weights.

    If problem is given, it must match the problem recorded in the checkpoint
    (checkpoints that record none are accepted). Returns (model, checkpoint), the
    checkpoint being the saved dict (history, settings, ...).
    '''
    ck= torch.load(ckpt_file)
    if problem is not None and ck.get('problem', problem) != problem:
        raise ValueError(f"{Path(ckpt_file).name} is for problem {ck['problem']!r}, not {problem!r}")

    model= build_committor(ck['model_params'], indicators=indicators)
    # build in the saved precision; load_state_dict would otherwise cast to the default dtype
    model.to(next(iter(ck['model'].values())).dtype)
    model.load_state_dict(ck['model'])
    return model, ck


def as_numpy(model):
    '''
    Wrap a model as a numpy function q(x): (..., d) -> (...), without gradients.
    '''
    dtype= next(model.parameters()).dtype
    def q(x):
        with torch.no_grad():
            return model(torch.as_tensor(np.asarray(x), dtype=dtype)).numpy()
    return q


def train(*, model, opt, sched, X, w_X, Y, w_Y, ckpt_file, model_params, hyperparams, info):
    '''
    Minibatch training of model on the points X (N, d) with loss weights w_X (N,),
    reporting the test loss on Y, w_Y after every epoch.

    hyperparams must contain n_epochs and batch_size; model_params, hyperparams and
    info (e.g. problem name and seed) are saved in the checkpoint, along with the
    model, optimizer, scheduler and torch RNG state and the loss history, after every
    epoch. If ckpt_file exists, training resumes from it: n_epochs may be raised to
    continue, but any other change in model_params or hyperparams raises an error.

    Returns the history dict (train_loss, test_loss and lr per epoch).
    '''
    history= dict(train_loss=[], test_loss=[], lr=[])
    start_epoch= 0
    if ckpt_file.exists():
        ck= torch.load(ckpt_file)
        saved_hp= {k: v for k, v in ck['hyperparams'].items() if k != 'n_epochs'}
        current_hp= {k: v for k, v in hyperparams.items() if k != 'n_epochs'}
        if ck['model_params'] != model_params or saved_hp != current_hp:
            raise ValueError(f'{ckpt_file.name} was trained with different settings; '
                             'change the run name or delete the checkpoint')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        torch.set_rng_state(ck['torch_rng'])
        history= ck['history']
        start_epoch= ck['epoch'] + 1
        print(f'resuming {ckpt_file.name} at epoch {start_epoch}')

    def save_checkpoint(epoch):
        ck= dict(model=model.state_dict(),
                 opt=opt.state_dict(),
                 sched=sched.state_dict(),
                 torch_rng=torch.get_rng_state(),
                 epoch=epoch,
                 history=history,
                 model_params=model_params,
                 hyperparams=hyperparams,
                 **info)
        # write to a temporary file first, so an interrupted save can't corrupt the checkpoint
        tmp= ckpt_file.with_suffix('.tmp')
        torch.save(ck, tmp)
        tmp.replace(ckpt_file)

    batch_size= hyperparams['batch_size']

    for epoch in range(start_epoch, hyperparams['n_epochs']):
        # one pass over the training set in a random order
        perm= torch.randperm(len(X))
        loss_sum= 0.0
        for i in range(0, len(X), batch_size):
            idx= perm[i:i + batch_size]
            loss= loss_fn(model=model, X=X[idx], weights=w_X[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * len(idx)

        history['lr'].append(sched.get_last_lr()[0])
        sched.step()

        # the training loss is averaged over the epoch while the weights change; the test
        # loss is evaluated once, at the end of the epoch
        history['train_loss'].append(loss_sum / len(X))
        history['test_loss'].append(loss_fn(model=model, X=Y, weights=w_Y, create_graph=False).item())

        save_checkpoint(epoch)
        print(f"epoch {epoch:4d}  train {history['train_loss'][-1]:.6e}  test {history['test_loss'][-1]:.6e}")

    return history
