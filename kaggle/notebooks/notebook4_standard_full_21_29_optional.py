import os
import subprocess

DATASET_ZIP = '/kaggle/input/smartmarl-codebase/smartmarl_kaggle.zip'
WORKDIR = '/kaggle/working'

subprocess.run(['apt-get', 'update', '-q'], check=False)
subprocess.run(['apt-get', 'install', '-y', 'sumo', 'sumo-tools'], check=True)
os.environ['SUMO_HOME'] = '/usr/share/sumo'
subprocess.run(['unzip', '-q', '-o', DATASET_ZIP, '-d', WORKDIR], check=True)

smoke = subprocess.run(
    ['python', 'train.py', '--scenario', 'standard', '--seed', '0', '--episodes', '1'],
    cwd=WORKDIR,
    capture_output=True,
    text=True,
)
assert 'Mock mode: False' in smoke.stdout, 'MOCK MODE DETECTED'

os.makedirs(f'{WORKDIR}/results/raw', exist_ok=True)

for seed in range(21, 30):
    cmd = [
        'python', 'train.py',
        '--scenario', 'standard',
        '--ablation', 'full',
        '--seed', str(seed),
        '--episodes', '3000',
        '--steps_per_episode', '300',
        '--checkpoint_every', '100',
        '--result_json', f'results/raw/standard_full_seed{seed}.json',
        '--resume',
    ]
    subprocess.run(cmd, cwd=WORKDIR, check=True)
    print(f'Seed {seed} complete')
