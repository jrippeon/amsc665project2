import torch
import numpy as np
import committor_model as cm
import mueller

# for filepaths
from pathlib import Path
ROOT= Path(__file__).resolve().parents[2]
data_dir= ROOT / 'data' / 'committor'

# the data are float64, and a model this small costs little in double precision
torch.set_default_dtype(torch.float64)

SEED= mueller.MUELLER_SEED
# torch seeds must fit in 64 bits, so derive one from SEED. Spawn key 1 keeps it
# independent of the test-set stream (key 0) in generate_training_set.py
torch.manual_seed(int(np.random.SeedSequence(SEED).spawn(2)[1].generate_state(1, dtype=np.uint64)[0]))

# name of this run; change it (or delete the checkpoint) to train with different settings
RUN_NAME= 'baseline'
ckpt_file= data_dir / f'committor_{RUN_NAME}_{SEED}.pt'

# model parameters. Everything here is saved in the checkpoint, so it must be plain
# numbers; chi_A and chi_B are rebuilt from r_chi and w_chi by mueller.indicators
beta= mueller.PARAMS['beta']

MODEL_PARAMS= dict(
    num_layers=3,
    hidden_size=10,
    d=10,
    r_chi=mueller.R,
    w_chi=0.01,
)

# training hyperparameters (Adam and the step schedule follow the Face_NN example)
HYPERPARAMS= dict(
    n_epochs=400,
    batch_size=256,
    lr=1e-3,
    milestones=[100, 200, 300],     # epochs at which the learning rate is multiplied by gamma
    gamma=0.5,
)

model= cm.build_committor(MODEL_PARAMS, indicators=mueller.indicators)
opt= torch.optim.Adam(model.parameters(), lr=HYPERPARAMS['lr'])
sched= torch.optim.lr_scheduler.MultiStepLR(opt, milestones=HYPERPARAMS['milestones'], gamma=HYPERPARAMS['gamma'])

# training and test data
X_np= np.load(data_dir / f'training_data_{SEED}.npy')
Y_np= np.load(data_dir / f'testing_data_{SEED}.npy')
X= torch.from_numpy(X_np)
Y= torch.from_numpy(Y_np)

# weights exp(-beta V) for quasi-uniform (delta-net) points. They don't depend on the
# model, so compute them once in numpy. Dividing by a fixed constant (the training mean)
# only rescales the loss, which keeps its printed values readable
w_X= np.exp(-beta * mueller.V(X_np))
w_Y= np.exp(-beta * mueller.V(Y_np))
scale= w_X.mean()
w_X= torch.from_numpy(w_X / scale)
w_Y= torch.from_numpy(w_Y / scale)

cm.train(model=model, opt=opt, sched=sched,
         X=X, w_X=w_X, Y=Y, w_Y=w_Y,
         ckpt_file=ckpt_file,
         model_params=MODEL_PARAMS,
         hyperparams=HYPERPARAMS,
         info=dict(problem='mueller', seed=SEED))
