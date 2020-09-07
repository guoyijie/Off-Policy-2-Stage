import os
import subprocess as sp

seeds = [i for i in range(20)]
kl_weights = [10]
decays = [0.9]
#kl_weights = [10, 1, 0.1]
#kl_weight_decays = [1]

jobs = []
for seed in seeds:
    for decay in decays:
        for kl_weight in kl_weights:
            log = "results/kl_weight_{}_decay_{}_s{}".format(kl_weight, decay, seed) 
            jobs.append({'seed':seed, 'kl_weight':kl_weight, 'decay':decay, 'log':log})


for job in jobs:
    print(job)

for job in jobs:
    path = job['log']
    if not os.path.exists(path):
        sp.call(['mkdir', path])
        print("Starting: ", job)
        sp.call(['python', 'train_off_continuation.py',
            '--seed', str(job['seed']),
            '--kl_weight', str(job['kl_weight']),
            '--decay', str(job['decay']),
            '--log', str(job['log'])])

