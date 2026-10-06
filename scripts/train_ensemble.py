"""Orchestrate N-seed x 5-member training with 5-way parallel.

Usage:
    python scripts/train_ensemble.py --env walker2d --seeds 0,1,2,3,4
    python scripts/train_ensemble.py --env hopper --seeds 5,6,7,8,9

Each seed runs its 5 members in parallel (one subprocess per member,
torch threads=2 each). Resume-safe: existing checkpoints are skipped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--env', required=True, choices=['hopper', 'walker2d'])
parser.add_argument('--seeds', required=True, help='comma-separated seed indices')
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--out_root', default='results/ensemble_training')
parser.add_argument('--force', action='store_true')
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(',')]
here = Path(__file__).resolve().parent
member_script = here / 'train_member.py'

print(f'=== Orchestrating {args.env} seeds {seeds} ===')
print(f'    threads per member: {args.threads}')
print(f'    out_root:           {args.out_root}')
print()

total_start = __import__('time').time()

for seed_idx in seeds:
    print(f'\n--- {args.env} seed {seed_idx} ---')
    t0 = __import__('time').time()
    procs = []
    for member_idx in range(5):
        cmd = [
            sys.executable, str(member_script),
            '--env', args.env,
            '--seed_idx', str(seed_idx),
            '--member_idx', str(member_idx),
            '--threads', str(args.threads),
            '--out_root', args.out_root,
        ]
        if args.force:
            cmd.append('--force')
        procs.append((member_idx, subprocess.Popen(cmd)))

    rcs = []
    for member_idx, p in procs:
        rc = p.wait()
        rcs.append(rc)
        status = 'OK' if rc == 0 else f'FAIL({rc})'
        print(f'  member {member_idx} {status}')
    print(f'  seed {seed_idx} total: {__import__("time").time()-t0:.1f}s')

total = __import__('time').time() - total_start
print()
print(f'=== DONE in {total:.1f}s = {total/60:.1f} min ===')