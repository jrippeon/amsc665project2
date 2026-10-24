import numpy as np
import metadynamics_optimized as md
import mueller
from pathlib import Path

ROOT= Path(__file__).resolve().parents[2]

SEED= mueller.MUELLER_SEED
rng= np.random.default_rng(SEED)
# independent stream for the test set; spawning leaves rng's own stream unchanged,
# and the child depends only on SEED, so it can be rebuilt without rerunning anything
rng_test= np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])

data_dir= ROOT / 'data' / 'committor'
data_dir.mkdir(parents=True, exist_ok=True)
centers_file= data_dir / f'centers_{SEED}.npy'

# (name, N_samples, rng): the training set uses rng right after deposition, the test set its own stream
sets= [('training', 210_000, rng),
       ('testing', 50_000, rng_test)]
missing= [s for s in sets if not (data_dir / f'{s[0]}_data_{SEED}.npy').exists()]

# deposit gaussian bumps. The training set's stream continues from the deposition, so
# rerun it whenever the training set is missing; otherwise the saved centers suffice
if centers_file.exists() and all(name != 'training' for name, _, _ in missing):
    x_centers= np.load(centers_file)
else:
    _,x_centers= md.metadynamics_deposition(dV=mueller.dV,
                                          cv=mueller.cv,
                                          dcv=mueller.dcv,
                                          rng=rng,
                                          x0=np.r_[mueller.CENTER_A, np.zeros(8)],   # start in basin A
                                          **mueller.PARAMS
                                          )
    # save intermediate results, so a later failure doesn't lose them
    np.save(centers_file, x_centers)

N_walkers= 100

for name, N_samples, set_rng in missing:
    # sample from the biased distribution
    x0= md.initialize_walkers(x_centers, N_walkers, rng=set_rng)
    x= md.metadynamics_sampling(N_samples=N_samples,
                                N_skip=100,
                                dV=mueller.dV,
                                x_centers=x_centers,
                                cv=mueller.cv,
                                dcv=mueller.dcv,
                                x0=x0,
                                rng=set_rng,
                                N_walkers=N_walkers,
                                **mueller.PARAMS
                               )
    np.save(data_dir / f'{name}_samples_{SEED}.npy', x)

    # the variational loss is over the complement of A and B, so drop points inside them
    x= x[~(mueller.in_A(x) | mueller.in_B(x))]

    # claude suspects Yuan et al did the delta masking in collective variable space
    # sounds good to me?
    mask= md.delta_net(mueller.cv(x), 0.005)

    np.save(data_dir / f'{name}_data_{SEED}.npy', x[mask])
