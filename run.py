"""Portable CLI; no dataset or checkpoint is selected implicitly."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def execute(stage, entry, args, gpu):
    env = os.environ.copy()
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['PYTHONUNBUFFERED'] = '1'
    subprocess.run([sys.executable, '-B', '-u', str(ROOT/stage/'newCodes'/entry),
                    '--gpu', str(gpu), *args], cwd=ROOT/stage/'newCodes',
                   env=env, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('train')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--resume', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--max-epoch', type=int, default=140)
    train.add_argument('--seed', type=int, default=2026)
    train.add_argument('--batch-size', type=int, default=4)
    train.add_argument('--workers', type=int, default=4)
    train.add_argument('--gpu', default='0')
    stitch = sub.add_parser('stitch')
    stitch.add_argument('--data', type=Path, required=True)
    stitch.add_argument('--warp-checkpoint', type=Path, required=True)
    stitch.add_argument('--fusion-checkpoint', type=Path, required=True)
    stitch.add_argument('--output', type=Path, required=True)
    stitch.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.output.exists():
        parser.error('Output already exists. Use a new folder to protect previous results.')
    required = ['input1', 'input2']
    if args.command == 'train':
        required += ['depthInput1', 'depthInput2']
    sets = []
    for name in required:
        files = sorted(p.name for p in (args.data/name).glob('*.jpg'))
        if not files:
            parser.error(f'No JPEG files: {args.data/name}')
        sets.append(files)
    if not all(s == sets[0] for s in sets):
        parser.error('Paired folders must have matching JPEG filenames')
    paths = [args.resume] if args.command == 'train' else [args.warp_checkpoint, args.fusion_checkpoint]
    for p in paths:
        if not p.is_file():
            parser.error(f'Missing checkpoint: {p}')
    if args.command == 'train':
        execute('Warp', 'train.py', [
            '--train_path', str(args.data), '--resume_path', str(args.resume.resolve()),
            '--experiment_dir', str(args.output), '--max_epoch', str(args.max_epoch),
            '--seed', str(args.seed), '--batch_size', str(args.batch_size),
            '--num_workers', str(args.workers), '--save_every', '5',
            '--depth_input_mode', 'correct', '--depth_loss_mode', 'true_depth',
            '--depth_gradient_projection', '1', '--depth_reliability_gate', '1',
            '--sample_information_gate', '1'], args.gpu)
    else:
        execute('Warp', 'test_output.py', [
            '--test_path', str(args.data), '--checkpoint', str(args.warp_checkpoint.resolve()),
            '--output_dir', str(args.output/'warp')], args.gpu)
        execute('Fusion', 'test.py', [
            '--test_path', str(args.output/'warp'), '--checkpoint', str(args.fusion_checkpoint.resolve()),
            '--output_dir', str(args.output/'fusion')], args.gpu)

if __name__ == '__main__':
    main()
