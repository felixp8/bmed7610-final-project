import os
import subprocess
import time
from datetime import datetime

cpu_list = list(range(0,64,4))
max_cpus = len(cpu_list)

neuron_list = list(range(0, 430))
# failed_neurons = [261, 267]
# neuron_list = failed_neurons + neuron_list
# done_neurons = [9, 18]
# neuron_list = [n for n in neuron_list if n not in done_neurons]

procs = {}

while len(neuron_list) > 0 or len(cpu_list) < max_cpus:
    for cpu, (n, proc) in procs.items():
        retcode = proc.poll()
        if retcode is not None:
            print(f'{datetime.now()}: CPU {cpu} done with neuron {n}')
            stdout, stderr = proc.communicate()
            stdout = str(stdout, "utf-8")
            stderr = str(stderr, "utf-8")
            with open('../analysis/mlp_temp.txt', 'a') as f:
                f.write(f'{cpu}\n{stdout}\n{stderr}\n')
            if 'Error' in stderr:
                print(f'{datetime.now()}: Error detected, re-running neuron {n}')
                neuron_list.append(n)
            cpu_list.append(cpu)
    for cpu in cpu_list:
        if cpu in procs:
            val = procs[cpu]
            if isinstance(val, tuple) and len(val) == 2:
                assert procs[cpu][1].poll() is not None
            del procs[cpu]
    while len(cpu_list) > 0 and len(neuron_list) > 0:
        run_cpu =  cpu_list.pop(0)
        run_neuron = neuron_list.pop(0)

        print(f'{datetime.now()}: Running neuron {run_neuron} on CPU {run_cpu}')
        p = subprocess.Popen(
            ["taskset", "--cpu-list", str(run_cpu) + '-' + str(run_cpu+3), "python", "glif_eval.py", str(run_neuron), str(run_neuron + 1)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        procs[run_cpu] = (run_neuron, p)
    
    if len(neuron_list) == 0 and len(cpu_list) == max_cpus:
        print('Run done')
        break

    time.sleep(30)