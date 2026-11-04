"""One real-data, RAM-only optimizer step for DSFN-L; writes no checkpoint."""
import argparse
import json
import os
from pathlib import Path
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--train-path', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'Warp/newCodes'))
    os.environ.setdefault('TORCH_HOME', str(root / 'torch_cache'))

    import numpy as np
    import torch
    from dataset import TrainDataset
    from depth_gradient import DepthGradientProjector
    from loss import (cal_information_reliable_true_depth_loss, cal_lp_loss,
                      inter_grid_loss, intra_grid_loss)
    from network import Network, build_train_model

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = TrainDataset(str(args.train_path))
    samples = [dataset[index] for index in range(args.batch_size)]
    batch = [torch.stack([sample[key] for sample in samples]).float().cuda()
             for key in range(4)]
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint.get('epoch') != 100:
        raise RuntimeError('Expected the shared epoch100 checkpoint')
    net = Network().cuda().train()
    net.load_state_dict(checkpoint['model'], strict=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4,
                                 betas=(0.9, 0.999), eps=1e-8)
    optimizer.load_state_dict(checkpoint['optimizer'])
    projector = DepthGradientProjector(net)
    del checkpoint

    before = next(net.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    out = build_train_model(net, *batch)
    rgb = cal_lp_loss(batch[0], batch[1], out['output_H'], out['output_H_inv'],
                      out['warp_mesh'], out['warp_mesh_mask'])
    grid = (10 * inter_grid_loss(out['overlap'], out['mesh2']) +
            10 * intra_grid_loss(out['mesh2']))
    (depth_raw, reliability, sample_gate, luminance_gate, structure_gate,
     luminance, structure) = cal_information_reliable_true_depth_loss(
        batch[2], batch[3], out['depth_H_2to1'], out['depth_H_2to1_mask'],
        out['depth_H_1to2'], out['depth_H_1to2_mask'], out['depth_tps_2to1'],
        out['depth_tps_2to1_mask'], batch[0], batch[1], out['output_H'],
        out['output_H_inv'], out['warp_mesh'])
    depth = 0.3 * depth_raw
    stats = projector.backward(rgb + grid, depth, enabled=True)
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0,
                                          error_if_nonfinite=True)
    optimizer.step()
    if torch.equal(before, next(net.parameters())):
        raise RuntimeError('optimizer did not update parameters')
    if not all(torch.isfinite(parameter).all() for parameter in net.parameters()):
        raise RuntimeError('non-finite parameter after optimizer step')
    report = {
        'passed': True,
        'variant': 'L',
        'batch_size': args.batch_size,
        'gpu': args.gpu,
        'start_epoch': 100,
        'rgb_loss': float(rgb.detach()),
        'grid_loss': float(grid.detach()),
        'depth_loss_weighted': float(depth.detach()),
        'spatial_reliability_mean': float(reliability.detach()),
        'sample_information_gate_mean': float(sample_gate.detach()),
        'sample_luminance_gate_mean': float(luminance_gate.detach()),
        'sample_structure_gate_mean': float(structure_gate.detach()),
        'pair_luminance_mean': float(luminance.detach()),
        'pair_structure_mean': float(structure.detach()),
        'gradient_norm_before_clip': float(norm),
        'gradient_stats': stats,
        'note': 'One real-data RAM-only optimizer step; no checkpoint written.'
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('PREFLIGHT PASS', args.output, flush=True)


if __name__ == '__main__':
    main()
