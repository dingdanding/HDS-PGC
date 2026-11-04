# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
from network import build_model, Network, RepConvN
from dataset import *
import os
import numpy as np
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
last_path = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(last_path)
DEFAULT_TEST_PATH = os.path.join(PROJECT_ROOT, 'UDIS-D', 'testing') + os.sep
MODEL_DIR = os.path.join(last_path, 'model')



def test(args):

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()

        
    # transform for deployment
    for m in net.modules():
        if isinstance(m, RepConvN):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward 
    
    #load the existing models if it exists
    if args.batch_size != 1:
        raise ValueError('Variable-size stitching output requires batch_size=1')
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    net.load_state_dict(checkpoint['model'], strict=True)
    print('Loaded explicitly selected checkpoint:', args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    path_learn_mask1 = os.path.join(args.output_dir, 'learn_mask1')
    if not os.path.exists(path_learn_mask1):
        os.makedirs(path_learn_mask1)
    path_learn_mask2 = os.path.join(args.output_dir, 'learn_mask2')
    if not os.path.exists(path_learn_mask2):
        os.makedirs(path_learn_mask2)
    path_final_composition = os.path.join(args.output_dir, 'composition')
    if not os.path.exists(path_final_composition):
        os.makedirs(path_final_composition)


    print("##################start testing#######################")
    net.eval()
    for i, batch_value in enumerate(test_loader):

        warp1_tensor = batch_value[0].float()
        warp2_tensor = batch_value[1].float()
        mask1_tensor = batch_value[2].float()
        mask2_tensor = batch_value[3].float()

        if torch.cuda.is_available():
            warp1_tensor = warp1_tensor.cuda()
            warp2_tensor = warp2_tensor.cuda()
            mask1_tensor = mask1_tensor.cuda()
            mask2_tensor = mask2_tensor.cuda()

        # if inpu1_tesnor.size()[2]*inpu1_tesnor.size()[3] > 1200000:
        #     print("oversize")
        #     continue

            # transform for deployment
        for m in net.modules():
            if isinstance(m, RepConvN):
                m.fuse_convs()
                m.forward = m.forward_fuse  # update forward
        
        
        with torch.no_grad():
            batch_out = build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

        stitched_image = batch_out['stitched_image']
        learned_mask1 = batch_out['learned_mask1']
        learned_mask2 = batch_out['learned_mask2']

        stitched_image = ((stitched_image[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        learned_mask1 = (learned_mask1[0]*255).cpu().detach().numpy().transpose(1,2,0)
        learned_mask2 = (learned_mask2[0]*255).cpu().detach().numpy().transpose(1,2,0)

        path = os.path.join(path_learn_mask1, str(i+1).zfill(6) + ".jpg")
        cv2.imwrite(path, learned_mask1)
        path = os.path.join(path_learn_mask2, str(i+1).zfill(6) + ".jpg")
        cv2.imwrite(path, learned_mask2)
        path = os.path.join(path_final_composition, str(i+1).zfill(6) + ".jpg")
        cv2.imwrite(path, stitched_image)


        print('i = {}'.format( i+1))



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
