import os
import subprocess

DATASET_ZIP = '/kaggle/input/smartmarl-codebase/smartmarl_kaggle.zip'
WORKDIR = '/kaggle/working'

print('Installing SUMO...')
subprocess.run(['apt-get', 'update', '-q'], check=False)
subprocess.run(['apt-get', 'install', '-y', 'sumo', 'sumo-tools'], check=True)
os.environ['SUMO_HOME'] = '/usr/share/sumo'

print('Unzipping code...')
subprocess.run(['unzip', '-q', '-o', DATASET_ZIP, '-d', WORKDIR], check=True)

print('Smoke test (must be real SUMO)...')
smoke = subprocess.run(
    ['python', 'train.py', '--scenario', 'standard', '--ablation', 'l7', '--seed', '0', '--episodes', '1'],
    cwd=WORKDIR,
    capture_output=True,
    text=True,
)
print(smoke.stdout[-2500:])
assert 'Mock mode: False' in smoke.stdout, 'MOCK MODE DETECTED: stop and fix SUMO setup'
print('Real SUMO confirmed.')

os.makedirs(f'{WORKDIR}/results/raw', exist_ok=True)

for seed in range(1, 30):
    print(f'Running standard/l7 seed {seed}')
    cmd = [
        'python', 'train.py',
        '--scenario', 'standard',
        '--ablation', 'l7',
        '--seed', str(seed),
        '--episodes', '3000',
        '--steps_per_episode', '300',
        '--checkpoint_every', '100',
        '--result_json', f'results/raw/standard_l7_seed{seed}.json',
        '--resume',
    ]
    subprocess.run(cmd, cwd=WORKDIR, check=True)
    print(f'Seed {seed} complete')

print('Notebook 3 done.')
