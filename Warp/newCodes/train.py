import argparse
import json
import random
from datetime import datetime
import time
from depth_gradient import DepthGradientProjector, GROUPS
import torch
from torch.utils.data import DataLoader
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from network import build_model, build_train_model, Network
from dataset import TrainDataset
from loss import (
    cal_lp_loss,
    inter_grid_loss,
    intra_grid_loss,
    cal_depth_loss,
    cal_true_depth_loss,
    cal_reliable_true_depth_loss,
    cal_information_reliable_true_depth_loss,
)
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
last_path = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(last_path)
DEFAULT_TRAIN_PATH = os.path.join(PROJECT_ROOT, 'UDIS-D', 'training') + os.sep


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_path_inside(path, root):
    path = os.path.normcase(os.path.abspath(path))
    root = os.path.normcase(os.path.abspath(root))
    return os.path.commonpath([path, root]) == root

def train(args):

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    train_path = os.path.abspath(args.train_path)
    experiment_dir = os.path.abspath(args.experiment_dir)
    if args.depth_input_mode != 'correct' or args.depth_loss_mode != 'true_depth':
        raise ValueError('L requires correct-depth inputs and true_depth loss')
    if not args.depth_reliability_gate:
        raise ValueError('L requires depth_reliability_gate=1')
    if not args.sample_information_gate:
        raise ValueError('L requires sample_information_gate=1')
    if args.depth_boundary_alpha < 0.0 or args.depth_structure_alpha < 0.0:
        raise ValueError('depth reliability alpha values must be non-negative')
    if not 0.0 <= args.depth_reliability_min <= 1.0:
        raise ValueError('depth_reliability_min must be in [0, 1]')
    if args.sample_luminance_full <= args.sample_luminance_low:
        raise ValueError('sample_luminance_full must exceed sample_luminance_low')
    if args.sample_structure_full <= args.sample_structure_low:
        raise ValueError('sample_structure_full must exceed sample_structure_low')
    if not 0.0 <= args.sample_information_min <= 1.0:
        raise ValueError('sample_information_min must be in [0, 1]')
    if args.num_workers < 0:
        raise ValueError('num_workers must be zero or greater')
    if args.save_every <= 0:
        raise ValueError('save_every must be greater than zero')
    if args.print_every <= 0:
        raise ValueError('print_every must be greater than zero')
    if not os.path.isdir(train_path):
        raise FileNotFoundError('training dataset not found: {}'.format(train_path))
    if not is_path_inside(experiment_dir, PROJECT_ROOT):
        raise ValueError('experiment_dir must stay inside {}: {}'.format(PROJECT_ROOT, experiment_dir))

    model_dir = os.path.join(experiment_dir, 'model')
    summary_dir = os.path.join(experiment_dir, 'summary')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    seed_everything(args.seed)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    writer = SummaryWriter(log_dir=summary_dir)

    print('project root: {}'.format(PROJECT_ROOT))
    print('experiment variant: L; grouped depth gradient projection: {}'.format(args.depth_gradient_projection))
    print('depth reliability gate: {} boundary_alpha={} structure_alpha={} minimum_weight={}'.format(
        args.depth_reliability_gate, args.depth_boundary_alpha,
        args.depth_structure_alpha, args.depth_reliability_min))
    print('sample information gate: {} luminance={}->{} structure={}->{} minimum_weight={}'.format(
        args.sample_information_gate, args.sample_luminance_low,
        args.sample_luminance_full, args.sample_structure_low,
        args.sample_structure_full, args.sample_information_min))
    print('experiment dir: {}'.format(experiment_dir))
    print('depth input mode: {}'.format(args.depth_input_mode))
    print('depth loss mode: {}'.format(args.depth_loss_mode))
    print('random seed: {}'.format(args.seed))

    config = vars(args).copy()
    config.update({
        'experiment_variant': 'L',
        'gradient_groups': list(GROUPS),
        'project_root': PROJECT_ROOT,
        'train_path': train_path,
        'experiment_dir': experiment_dir,
        'resume_path': os.path.abspath(args.resume_path) if args.resume_path else '',
    })
    with open(os.path.join(experiment_dir, 'run_config.json'), 'w', encoding='utf-8') as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=2)

    # define dataset
    train_data = TrainDataset(data_path=train_path)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
    projector = DepthGradientProjector(net)

    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # Resume only from an explicitly selected checkpoint. This prevents an
    # experiment from silently loading or overwriting another experiment.
    if args.resume_path:
        model_path = os.path.abspath(args.resume_path)
        if not os.path.isfile(model_path):
            writer.close()
            raise FileNotFoundError('checkpoint not found: {}'.format(model_path))
        checkpoint = torch.load(model_path, weights_only=False)
        if checkpoint.get('experiment_variant') not in (None, 'L'):
            raise ValueError('Cannot resume a different experiment variant')
        if checkpoint.get('experiment_variant') is None and checkpoint['epoch'] != 100:
            raise ValueError('Controlled L training must start at the shared epoch100 checkpoint')
        if checkpoint.get('experiment_variant') == 'L' and checkpoint.get('depth_gradient_projection') != args.depth_gradient_projection:
            raise ValueError('Do not change projection mode when resuming')
        if checkpoint.get('experiment_variant') == 'L':
            expected_reliability = {
                'depth_reliability_gate': args.depth_reliability_gate,
                'depth_boundary_alpha': args.depth_boundary_alpha,
                'depth_structure_alpha': args.depth_structure_alpha,
                'depth_reliability_min': args.depth_reliability_min,
                'sample_information_gate': args.sample_information_gate,
                'sample_luminance_low': args.sample_luminance_low,
                'sample_luminance_full': args.sample_luminance_full,
                'sample_structure_low': args.sample_structure_low,
                'sample_structure_full': args.sample_structure_full,
                'sample_information_min': args.sample_information_min,
            }
            for key, value in expected_reliability.items():
                if checkpoint.get(key) != value:
                    raise ValueError('Do not change {} when resuming L'.format(key))
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
        # GPU optimizer/model state is now owned by the model and optimizer.
        # Release the checkpoint dictionary before the forward pass.
        del checkpoint
    else:
        existing_models = [name for name in os.listdir(model_dir) if name.endswith('.pth')]
        if existing_models:
            writer.close()
            raise RuntimeError(
                'experiment model directory is not empty; select --resume_path explicitly: {}'.format(model_dir)
            )
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')

    if args.max_epoch <= start_epoch:
        writer.close()
        raise ValueError(
            'max_epoch ({}) must be greater than checkpoint epoch ({})'.format(
                args.max_epoch, start_epoch
            )
        )

    print("##################start training#######################")
    config.update(start_epoch=start_epoch, initial_lr=optimizer.param_groups[0]['lr'])
    with open(os.path.join(experiment_dir, 'run_config.json'), 'w', encoding='utf-8') as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2)
    score_print_fre = args.print_every

    for epoch in range(start_epoch, args.max_epoch):

        print("start epoch {}".format(epoch))
        net.train()
        loss_sigma = 0.0
        overlap_loss_sigma = 0.
        nonoverlap_loss_sigma = 0.
        overlap_depth_loss_sigma = 0.
        depth_reliability_sigma = 0.
        sample_information_sigma = 0.
        sample_luminance_gate_sigma = 0.
        sample_structure_gate_sigma = 0.
        pair_luminance_sigma = 0.
        pair_structure_sigma = 0.
        logged_batches = 0
        conflict_sigma = {g: 0.0 for g in GROUPS}
        cosine_sigma = {g: 0.0 for g in GROUPS}
        ratio_sigma = {g: 0.0 for g in GROUPS}
        window_start = time.perf_counter()

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            inpu1_tesnor = batch_value[0].float()
            inpu2_tesnor = batch_value[1].float()
            depthInput1_tensor = batch_value[2].float()
            depthInput2_tensor = batch_value[3].float()
            if torch.cuda.is_available():
                inpu1_tesnor = inpu1_tesnor.cuda()
                inpu2_tesnor = inpu2_tesnor.cuda()
                depthInput1_tensor = depthInput1_tensor.cuda()
                depthInput2_tensor = depthInput2_tensor.cuda()

            if args.depth_input_mode == 'legacy':
                depthInput2_for_loss = depthInput1_tensor
            else:
                depthInput2_for_loss = depthInput2_tensor

            # forward, backward, update weights
            optimizer.zero_grad(set_to_none=True)

            if args.depth_loss_mode == 'true_depth':
                batch_out = build_train_model(
                    net,
                    inpu1_tesnor,
                    inpu2_tesnor,
                    depthInput1_tensor,
                    depthInput2_tensor,
                )
            else:
                batch_out = build_model(net, inpu1_tesnor, inpu2_tesnor)
            # result
            output_H = batch_out['output_H']
            output_H_inv = batch_out['output_H_inv']
            warp_mesh = batch_out['warp_mesh']
            warp_mesh_mask = batch_out['warp_mesh_mask']
            mesh1 = batch_out['mesh1']
            mesh2 = batch_out['mesh2']
            overlap = batch_out['overlap']
#             depth_warp1 = batch_out['depth_warp1']
#             depth_warp2 = batch_out['depth_warp2']
#             depth_mask1 = batch_out['depth_mask1']
#             depth_mask2 = batch_out['depth_mask2']
            
            
            # calculate loss for overlapping regions
            overlap_loss = cal_lp_loss(inpu1_tesnor, inpu2_tesnor, output_H, output_H_inv, warp_mesh, warp_mesh_mask)
            # calculate loss for non-overlapping regions
            nonoverlap_loss = 10*inter_grid_loss(overlap, mesh2) + 10*intra_grid_loss(mesh2)
            
            # ----- add ----- calculate overlap depth loss
#             overlap_depth_loss = depth_loss(depth_warp1, depth_warp2, depth_mask1, depth_mask2)
            if args.depth_loss_mode == 'true_depth':
                (overlap_depth_loss, depth_reliability_mean,
                 sample_information_mean, sample_luminance_gate_mean,
                 sample_structure_gate_mean, pair_luminance_mean,
                 pair_structure_mean) = cal_information_reliable_true_depth_loss(
                    depthInput1_tensor,
                    depthInput2_tensor,
                    batch_out['depth_H_2to1'],
                    batch_out['depth_H_2to1_mask'],
                    batch_out['depth_H_1to2'],
                    batch_out['depth_H_1to2_mask'],
                    batch_out['depth_tps_2to1'],
                    batch_out['depth_tps_2to1_mask'],
                    inpu1_tesnor,
                    inpu2_tesnor,
                    output_H,
                    output_H_inv,
                    warp_mesh,
                    boundary_alpha=args.depth_boundary_alpha,
                    structure_alpha=args.depth_structure_alpha,
                    minimum_weight=args.depth_reliability_min,
                    sample_luminance_low=args.sample_luminance_low,
                    sample_luminance_full=args.sample_luminance_full,
                    sample_structure_low=args.sample_structure_low,
                    sample_structure_full=args.sample_structure_full,
                    sample_minimum_weight=args.sample_information_min,
                )
            else:
                overlap_depth_loss = cal_depth_loss(
                    depthInput1_tensor,
                    depthInput2_for_loss,
                    output_H,
                    output_H_inv,
                    warp_mesh,
                    warp_mesh_mask,
                )
                depth_reliability_mean = overlap_depth_loss.new_tensor(1.0)
                sample_information_mean = overlap_depth_loss.new_tensor(1.0)
                sample_luminance_gate_mean = overlap_depth_loss.new_tensor(1.0)
                sample_structure_gate_mean = overlap_depth_loss.new_tensor(1.0)
                pair_luminance_mean = overlap_depth_loss.new_tensor(1.0)
                pair_structure_mean = overlap_depth_loss.new_tensor(1.0)
            # conf
            overlap_depth_loss = 0.3 * overlap_depth_loss
            

            total_loss = overlap_loss +  nonoverlap_loss +  overlap_depth_loss
            # Total Loss remains a reporting quantity; with projection enabled,
            # the actual update is the explicit grouped gradient combination.
            gradient_stats = projector.backward(
                overlap_loss + nonoverlap_loss, overlap_depth_loss,
                enabled=bool(args.depth_gradient_projection))
            for group in GROUPS:
                if gradient_stats:
                    conflict_sigma[group] += gradient_stats[group]['conflict']
                    cosine_sigma[group] += gradient_stats[group]['cosine']
                    ratio_sigma[group] += gradient_stats[group]['depth_main_norm_ratio']
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2, error_if_nonfinite=True)
            optimizer.step()

            overlap_loss_sigma += overlap_loss.item()
            nonoverlap_loss_sigma += nonoverlap_loss.item()
            overlap_depth_loss_sigma += overlap_depth_loss.item()
            depth_reliability_sigma += depth_reliability_mean.item()
            sample_information_sigma += sample_information_mean.item()
            sample_luminance_gate_sigma += sample_luminance_gate_mean.item()
            sample_structure_gate_sigma += sample_structure_gate_mean.item()
            pair_luminance_sigma += pair_luminance_mean.item()
            pair_structure_sigma += pair_structure_mean.item()
            loss_sigma += total_loss.item()
            logged_batches += 1

#             print(glob_iter)

            # record loss and images in tensorboard
            if i % score_print_fre == 0 and i != 0:
                average_loss = loss_sigma / logged_batches
                average_overlap_loss = overlap_loss_sigma / logged_batches
                average_nonoverlap_loss = nonoverlap_loss_sigma / logged_batches
                average_overlap_depth_loss = overlap_depth_loss_sigma / logged_batches
                average_depth_reliability = depth_reliability_sigma / logged_batches
                average_sample_information = sample_information_sigma / logged_batches
                average_luminance_gate = sample_luminance_gate_sigma / logged_batches
                average_structure_gate = sample_structure_gate_sigma / logged_batches
                average_pair_luminance = pair_luminance_sigma / logged_batches
                average_pair_structure = pair_structure_sigma / logged_batches
                seconds_per_batch = (time.perf_counter()-window_start) / logged_batches
                print('[{}] variant=L projection={} pixel_reliability={:.3f} sample_gate={:.3f} luminance_gate={:.3f} structure_gate={:.3f} luminance={:.3f} structure={:.4f} seconds/batch={:.3f} conflicts(backbone/global/local)={}'.format(
                    datetime.now().isoformat(timespec='seconds'), args.depth_gradient_projection,
                    average_depth_reliability, average_sample_information,
                    average_luminance_gate, average_structure_gate,
                    average_pair_luminance, average_pair_structure, seconds_per_batch,
                    '/'.join('{:.3f}'.format(conflict_sigma[g]/logged_batches) for g in GROUPS)), flush=True)
                for group in GROUPS:
                    writer.add_scalar('gradient_conflict/'+group, conflict_sigma[group]/logged_batches, glob_iter)
                    writer.add_scalar('gradient_cosine/'+group, cosine_sigma[group]/logged_batches, glob_iter)
                    writer.add_scalar('gradient_norm_ratio/'+group, ratio_sigma[group]/logged_batches, glob_iter)
                    conflict_sigma[group] = cosine_sigma[group] = ratio_sigma[group] = 0.0
                writer.add_scalar('seconds_per_batch', seconds_per_batch, glob_iter)
                writer.add_scalar('depth_reliability_mean', average_depth_reliability, glob_iter)
                writer.add_scalar('sample_information_gate_mean', average_sample_information, glob_iter)
                writer.add_scalar('sample_luminance_gate_mean', average_luminance_gate, glob_iter)
                writer.add_scalar('sample_structure_gate_mean', average_structure_gate, glob_iter)
                writer.add_scalar('pair_luminance_mean', average_pair_luminance, glob_iter)
                writer.add_scalar('pair_structure_mean', average_pair_structure, glob_iter)
                logged_batches = 0
                window_start = time.perf_counter()
                loss_sigma = 0.0
                overlap_loss_sigma = 0.
                nonoverlap_loss_sigma = 0.
                overlap_depth_loss_sigma = 0.
                depth_reliability_sigma = 0.
                sample_information_sigma = 0.
                sample_luminance_gate_sigma = 0.
                sample_structure_gate_sigma = 0.
                pair_luminance_sigma = 0.
                pair_structure_sigma = 0.
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} overlap_depth_loss:{:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader),
                                          average_loss, average_overlap_loss, average_nonoverlap_loss, average_overlap_depth_loss, optimizer.state_dict()['param_groups'][0]['lr']))
                # visualization
                writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', average_loss, glob_iter)
                writer.add_scalar('overlap loss', average_overlap_loss, glob_iter)
                writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, glob_iter)
                writer.add_scalar('depth_loss', average_overlap_depth_loss, glob_iter)

            glob_iter += 1


        scheduler.step()
        # Save each controlled experiment independently.
        if ((epoch+1) % args.save_every == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model_.pth'
            model_save_path = os.path.join(model_dir, filename)
            if os.path.exists(model_save_path):
                writer.close()
                raise FileExistsError('refuse to overwrite checkpoint: {}'.format(model_save_path))
            state = {
                'experiment_variant': 'L',
                'depth_gradient_projection': args.depth_gradient_projection,
                'gradient_groups': list(GROUPS),
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch+1,
                'glob_iter': glob_iter,
                'depth_input_mode': args.depth_input_mode,
                'depth_loss_mode': args.depth_loss_mode,
                'depth_reliability_gate': args.depth_reliability_gate,
                'depth_boundary_alpha': args.depth_boundary_alpha,
                'depth_structure_alpha': args.depth_structure_alpha,
                'depth_reliability_min': args.depth_reliability_min,
                'sample_information_gate': args.sample_information_gate,
                'sample_luminance_low': args.sample_luminance_low,
                'sample_luminance_full': args.sample_luminance_full,
                'sample_structure_low': args.sample_structure_low,
                'sample_structure_full': args.sample_structure_full,
                'sample_information_min': args.sample_information_min,
                'seed': args.seed,
                'resume_path': os.path.abspath(args.resume_path) if args.resume_path else '',
            }
            torch.save(state, model_save_path)
            print('save model to {}!'.format(model_save_path))
    writer.close()
    print("##################end training#######################")


if __name__=="__main__":


    print('<==================== setting arguments ===================>\n')

    # create the argument parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--train_path', type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument('--experiment_dir', type=str, required=True)
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--depth_input_mode', choices=['legacy', 'correct'], required=True)
    parser.add_argument(
        '--depth_loss_mode',
        choices=['legacy_rgb', 'true_depth'],
        default='legacy_rgb',
        help='legacy_rgb reproduces the released cross-modal loss; true_depth compares geometrically aligned depth maps.',
    )
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--depth_gradient_projection', type=int, choices=[0,1], default=1,
                        help='1: asymmetric grouped auxiliary projection; 0: exact C loss sum for diagnostics.')
    parser.add_argument('--depth_reliability_gate', type=int, choices=[0,1], default=1,
                        help='K spatial reliability gate retained by L; controlled L training requires 1.')
    parser.add_argument('--depth_boundary_alpha', type=float, default=0.5)
    parser.add_argument('--depth_structure_alpha', type=float, default=0.5)
    parser.add_argument('--depth_reliability_min', type=float, default=0.25)
    parser.add_argument('--sample_information_gate', type=int, choices=[0,1], default=1,
                        help='L pair-level luminance/structure gate; controlled L training requires 1.')
    parser.add_argument('--sample_luminance_low', type=float, default=0.22277420572936543)
    parser.add_argument('--sample_luminance_full', type=float, default=0.40093238055706026)
    parser.add_argument('--sample_structure_low', type=float, default=0.025516219437122345)
    parser.add_argument('--sample_structure_full', type=float, default=0.05054995194077492)
    parser.add_argument('--sample_information_min', type=float, default=0.25)

    # parse the arguments
    args = parser.parse_args()
    print(args)

    # train
    train(args)




