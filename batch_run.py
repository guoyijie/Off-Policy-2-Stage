import os
import subprocess as sp

seeds = [i for i in range(20)]

jobs = []
for seed in seeds:
    log = "results/ips_random_batch_s{}".format(seed) 
    jobs.append({'seed':seed, 'log':log})


for job in jobs:
    print(job)

for job in jobs:
    path = job['log']
    if not os.path.exists(path):
        sp.call(['mkdir', path])
        print("Starting: ", job)
        sp.call(['python', 'run.py',
            '--seed', str(job['seed']),
            '--log', str(job['log'])])
