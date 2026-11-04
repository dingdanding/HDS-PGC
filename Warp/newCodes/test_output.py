# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_output_model, Network
from dataset import *
import os
import cv2
import time

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
last_path = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(last_path)
DEFAULT_TEST_PATH = os.path.join(PROJECT_ROOT, 'UDIS-D', 'testing') + os.sep
MODEL_DIR = os.path.join(last_path, 'model')


def draw_mesh_on_warp(warp, f_local):


    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)

    return warp

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return




def test(args):

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.test_path = os.path.join(os.path.abspath(args.test_path), '')
    
    # dataset
    test_data = TestDataset(data_path=args.test_path)
    #nl: set num_workers = the number of cpus
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing models if it exists
    if args.batch_size != 1:
        raise ValueError('Variable-size stitching output requires batch_size=1')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    net.load_state_dict(checkpoint['model'], strict=True)
    print('Loaded explicitly selected checkpoint:', args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    print("##################start testing#######################")
    # create folders if it dose not exist

    path_ave_fusion = os.path.join(args.output_dir, 'ave_fusion') + os.sep
    if not os.path.exists(path_ave_fusion):
        os.makedirs(path_ave_fusion)
    path_warp1 = os.path.join(args.output_dir, 'warp1') + os.sep
    if not os.path.exists(path_warp1):
        os.makedirs(path_warp1)
    path_warp2 = os.path.join(args.output_dir, 'warp2') + os.sep
    if not os.path.exists(path_warp2):
        os.makedirs(path_warp2)
    path_mask1 = os.path.join(args.output_dir, 'mask1') + os.sep
    if not os.path.exists(path_mask1):
        os.makedirs(path_mask1)
    path_mask2 = os.path.join(args.output_dir, 'mask2') + os.sep
    if not os.path.exists(path_mask2):
        os.makedirs(path_mask2)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of trainable parameters:", total_params)
    start_time = time.perf_counter()

    net.eval()
    for i, batch_value in enumerate(test_loader):

        #if i != 975:
        #    continue

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            inpu1_tesnor = inpu1_tesnor.cuda()
            inpu2_tesnor = inpu2_tesnor.cuda()

        with torch.no_grad():
            batch_out = build_output_model(net, inpu1_tesnor, inpu2_tesnor)

        final_warp1 = batch_out['final_warp1']
        final_warp1_mask = batch_out['final_warp1_mask']
        final_warp2 = batch_out['final_warp2']
        final_warp2_mask = batch_out['final_warp2_mask']
        final_mesh1 = batch_out['mesh1']
        final_mesh2 = batch_out['mesh2']


        final_warp1 = ((final_warp1[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp2 = ((final_warp2[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        final_warp1_mask = final_warp1_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_warp2_mask = final_warp2_mask[0].cpu().detach().numpy().transpose(1,2,0)
        final_mesh1 = final_mesh1[0].cpu().detach().numpy()
        final_mesh2 = final_mesh2[0].cpu().detach().numpy()



        path = path_warp1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1)
        path = path_warp2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2)
        path = path_mask1 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp1_mask*255)
        path = path_mask2 + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, final_warp2_mask*255)

        ave_fusion = final_warp1 * (final_warp1/ (final_warp1+final_warp2+1e-6)) + final_warp2 * (final_warp2/ (final_warp1+final_warp2+1e-6))
        path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, ave_fusion)

        print('i = {}'.format( i+1))

        torch.cuda.empty_cache()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"执行时间：{elapsed_time}秒")


    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    
    parser.add_argument('--test_path', type=str, default=DEFAULT_TEST_PATH)


    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
