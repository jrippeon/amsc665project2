import numpy as np
import metadynamics_optimized as md
import mueller

SEED= 31415926535897932384626433
rng= np.random.default_rng(SEED)
prefix= f'data/committor/training_data_{SEED}'
filename= f'{prefix}.npy'



# deposite gaussian bumps
_,x_centers= md.metadynamics_deposition(dV=mueller.dV,
                                      cv=mueller.cv,
                                      dcv=mueller.dcv,
                                      rng=rng,
                                      x0=np.r_[mueller.CENTER_A, np.zeros(8)],   # start in basin A
                                      **mueller.PARAMS
                                      )
# save intermediate results, so a later failure doesn't lose them
np.save(f'{prefix}_centers.npy', x_centers)

# sample from the new distribution
N_walkers= 100
x0= md.initialize_walkers(x_centers, N_walkers, rng=rng)

x= md.metadynamics_sampling(N_samples=210_000,
                            N_skip=100,
                            dV=mueller.dV,
                            x_centers=x_centers,
                            cv=mueller.cv,
                            dcv=mueller.dcv,
                            x0=x0,
                            rng=rng,
                            N_walkers=N_walkers,
                            **mueller.PARAMS
                           )
np.save(f'{prefix}_samples.npy', x)

# the variational loss is over the complement of A and B, so drop points inside them
x= x[~(mueller.in_A(x) | mueller.in_B(x))]

# claude suspects Yuan et al did the delta masking in collective variable space
# sounds good to me?
mask= md.delta_net(mueller.cv(x), 0.005)

np.save(filename, x[mask])

