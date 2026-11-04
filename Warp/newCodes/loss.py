import torch
import torch.nn as nn
import torch.nn.functional as F

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def cal_lp_loss(input1, input2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    # part one: sym homo loss with color balance
    delta1 = ( torch.sum(output_H[:,0:3,:,:], [2,3])  -   torch.sum(input1*output_H[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H[:,3:6,:,:], [2,3])

    input1_balance = input1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = ( torch.sum(output_H_inv[:,0:3,:,:], [2,3])  -   torch.sum(input2*output_H_inv[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H_inv[:,3:6,:,:], [2,3])
    input2_balance = input2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_1 = l_num_loss(input1_balance*output_H[:,3:6,:,:], output_H[:,0:3,:,:], 1) + l_num_loss(input2_balance*output_H_inv[:,3:6,:,:], output_H_inv[:,0:3,:,:], 1)

    # part two: tps loss with color balance
    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)


    lp_loss = 3. * lp_loss_1 + 1. * lp_loss_2

    return lp_loss

def cal_lp_loss2(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()

    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(input1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = l_num_loss(input1_newbalance*warp_mesh_mask, warp_mesh, 1)
    lp_loss =  1. * lp_loss_2

    return lp_loss

def inter_grid_loss(overlap, mesh):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]
    ##############################

    # on overlapping regions
    depth_diff_w = (1-torch.abs(overlap[:,:,0:grid_w-1] - overlap[:,:,1:grid_w])) * overlap[:,:,0:grid_w-1]
    error_w = depth_diff_w * delta_w_angle
    # on overlapping regions
    depth_diff_h = (1-torch.abs(overlap[:,0:grid_h-1,:] - overlap[:,1:grid_h,:])) * overlap[:,0:grid_h-1,:]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)



# intra-grid constraint
def intra_grid_loss(pts):

    max_w = 512/grid_w * 2
    max_h = 512/grid_h * 2

    delta_x = pts[:,:,1:grid_w+1,0] - pts[:,:,0:grid_w,0]
    delta_y = pts[:,1:grid_h+1,:,1] - pts[:,0:grid_h,:,1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)


    return loss



# ----- add -----

def depth_loss(depth_warp1, depth_warp2, depth_mask1, depth_mask2):
    mask_overlap = depth_warp1 * depth_warp2
    depth_mask_overlap = torch.where(mask_overlap > 0, torch.ones_like(mask_overlap), torch.zeros_like(mask_overlap))
#     print("depth_mask1.shape:")
#     print(depth_mask1.shape)
#     print("depth_mask2.shape:")
#     print(depth_mask2.shape)
#     print("depth_mask_overlap.shape:")
#     print(depth_mask_overlap.shape)
#     print("depth_warp1.shape:")
#     print(depth_warp1.shape)
    masked_depth_warp1 = depth_warp1 * depth_mask_overlap
    masked_depth_warp2 = depth_warp2 * depth_mask_overlap
    ## Z分数归一化
    # min_val1 = torch.min(masked_depth_warp1[depth_mask_overlap == 1])
    # max_val1 = torch.max(masked_depth_warp1[depth_mask_overlap == 1])
    # min_val2 = torch.min(masked_depth_warp2[depth_mask_overlap == 1])
    # max_val2 = torch.max(masked_depth_warp1[depth_mask_overlap == 1])
    # min_depth = torch.min(min_val1, min_val2)
    depth_mean = (torch.mean(masked_depth_warp1[depth_mask_overlap == 1]) + torch.mean(masked_depth_warp2[depth_mask_overlap == 1])) / 2
    depth_std = (torch.std(masked_depth_warp1[depth_mask_overlap == 1]) + torch.std(masked_depth_warp2[depth_mask_overlap == 1])) / 2
    z_score_depth_warp1 = (masked_depth_warp1 - depth_mean) / depth_std
    z_score_depth_warp2 = (masked_depth_warp2 - depth_mean) / depth_std
    final_z_depth_1 = z_score_depth_warp1 * depth_mask_overlap
    final_z_depth_2 = z_score_depth_warp2 * depth_mask_overlap

    depth_loss = torch.abs(final_z_depth_1 - final_z_depth_2) ** 2
    
    return torch.mean(depth_loss)

def depth_num_loss(depth_img1, depth_img2, l_num=1):
    # Z-score standardization 
#     depth_mean = (torch.mean(depth_img1) + torch.mean(depth_img2)) / 2
#     depth_std = (torch.std(depth_img1) + torch.std(depth_img2)) / 2
#     z_score_depth1 = (depth_img1 - depth_mean) / depth_std
#     z_score_depth2 = (depth_img2 - depth_mean) / depth_std
#     return torch.mean(torch.abs((z_score_depth1 - z_score_depth2)**l_num))
    combined_depth_maps = torch.cat((depth_img1,depth_img2))
    min_score = torch.min(combined_depth_maps)
    max_score = torch.max(combined_depth_maps)

    new_depth1 = (depth_img1 - min_score) / (max_score - min_score)
    new_depth2 = (depth_img2 - min_score) / (max_score - min_score)
    return torch.mean(torch.abs((new_depth1 - new_depth2)**l_num))

def cal_depth_loss(depthInput1, depthInput2, output_H, output_H_inv, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = depthInput1.size()

    # part one: sym homo loss with color balance
    delta1 = ( torch.sum(output_H[:,0:3,:,:], [2,3])  -   torch.sum(depthInput1*output_H[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H[:,3:6,:,:], [2,3])
    depth_input1_balance = depthInput1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = ( torch.sum(output_H_inv[:,0:3,:,:], [2,3])  -   torch.sum(depthInput2*output_H_inv[:,3:6,:,:], [2,3]) ) /  torch.sum(output_H_inv[:,3:6,:,:], [2,3])
    depth_input2_balance = depthInput2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    depth_lp_loss_1 = depth_num_loss(depth_input1_balance*output_H[:,3:6,:,:], output_H[:,0:3,:,:], 1) + depth_num_loss(depth_input2_balance*output_H_inv[:,3:6,:,:], output_H_inv[:,0:3,:,:], 1)

    # part two: tps loss with color balance
    delta3 = ( torch.sum(warp_mesh, [2,3])  -   torch.sum(depthInput1*warp_mesh_mask, [2,3]) ) /  torch.sum(warp_mesh_mask, [2,3])
    depth_input1_newbalance = depthInput1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    depth_lp_loss_2 = depth_num_loss(depth_input1_newbalance*warp_mesh_mask, warp_mesh, 1)


    depth_lp_loss = 3. * depth_lp_loss_1 + 1. * depth_lp_loss_2

    return depth_lp_loss


def _masked_standardize(depth, mask, eps=1e-6):
    """Normalize relative depth independently inside each valid overlap."""
    count = torch.sum(mask, dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = torch.sum(depth * mask, dim=(2, 3), keepdim=True) / count
    variance = torch.sum(((depth - mean) ** 2) * mask, dim=(2, 3), keepdim=True) / count
    return (depth - mean) / torch.sqrt(variance + eps)


def _masked_depth_consistency(reference, warped, valid_mask, eps=1e-6):
    """Scale/shift-invariant L1 consistency for one aligned depth pair."""
    # Stop the network from reducing the loss by shrinking the soft overlap.
    mask = valid_mask.clamp(0.0, 1.0).detach()
    reference_normalized = _masked_standardize(reference, mask, eps)
    warped_normalized = _masked_standardize(warped, mask, eps)
    difference = torch.abs(reference_normalized - warped_normalized) * mask
    count = torch.sum(mask, dim=(1, 2, 3)).clamp_min(1.0)
    return torch.mean(torch.sum(difference, dim=(1, 2, 3)) / count)


def cal_true_depth_loss(
        depthInput1,
        depthInput2,
        depth_H_2to1,
        depth_H_2to1_mask,
        depth_H_1to2,
        depth_H_1to2_mask,
        depth_tps_2to1,
        depth_tps_2to1_mask):
    """Compare depth with depth after reusing the RGB-predicted geometry."""
    homography_loss = (
        _masked_depth_consistency(depthInput1, depth_H_2to1, depth_H_2to1_mask)
        + _masked_depth_consistency(depthInput2, depth_H_1to2, depth_H_1to2_mask)
    )
    tps_loss = _masked_depth_consistency(
        depthInput1,
        depth_tps_2to1,
        depth_tps_2to1_mask,
    )
    # Keep the released 3:1 global/local weighting. On real UDIS-D samples
    # this remains close to the numeric scale of the original depth term.
    return 3.0 * homography_loss + tps_loss


def _gradient_magnitude(image, eps=1e-6):
    """First-order image gradient with the original spatial resolution."""
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
    dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
    return torch.sqrt(dx ** 2 + dy ** 2 + eps)


def _to_gray(image):
    if image.size(1) == 1:
        return image
    if image.size(1) != 3:
        raise ValueError('reliability guidance expects one- or three-channel images')
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return torch.sum(image * weights, dim=1, keepdim=True)


def _masked_normalize_positive(value, mask, eps=1e-6):
    count = torch.sum(mask, dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = torch.sum(value * mask, dim=(2, 3), keepdim=True) / count
    return (value / (mean + eps)).clamp(0.0, 6.0)


def _depth_reliability(
        reference_depth,
        warped_depth,
        reference_rgb,
        warped_rgb,
        valid_mask,
        boundary_alpha,
        structure_alpha,
        minimum_weight,
        eps=1e-6):
    """Build a detached soft confidence map from depth/RGB structure.

    Strong depth discontinuities are treated conservatively because resampling
    across object boundaries is ambiguous. A depth edge is additionally
    discounted when it is not supported by an RGB edge in the same view. RGB
    texture without a depth edge is not penalized, so repeated brick/window
    patterns can still benefit from a smooth depth surface.
    """
    mask = valid_mask.clamp(0.0, 1.0).detach()
    reference_depth = reference_depth.detach()
    warped_depth = warped_depth.detach()
    reference_rgb = reference_rgb.detach()
    warped_rgb = warped_rgb.detach()
    reference_depth = _masked_standardize(reference_depth, mask, eps)
    warped_depth = _masked_standardize(warped_depth, mask, eps)
    reference_gray = _masked_standardize(_to_gray(reference_rgb), mask, eps)
    warped_gray = _masked_standardize(_to_gray(warped_rgb), mask, eps)

    reference_depth_edge = _masked_normalize_positive(
        _gradient_magnitude(reference_depth, eps), mask, eps)
    warped_depth_edge = _masked_normalize_positive(
        _gradient_magnitude(warped_depth, eps), mask, eps)
    reference_rgb_edge = _masked_normalize_positive(
        _gradient_magnitude(reference_gray, eps), mask, eps)
    warped_rgb_edge = _masked_normalize_positive(
        _gradient_magnitude(warped_gray, eps), mask, eps)

    boundary = torch.maximum(reference_depth_edge, warped_depth_edge)
    unsupported_depth_edge = 0.5 * (
        F.relu(reference_depth_edge - reference_rgb_edge)
        + F.relu(warped_depth_edge - warped_rgb_edge)
    )
    boundary_confidence = 1.0 / (1.0 + boundary_alpha * boundary)
    structure_confidence = 1.0 / (
        1.0 + structure_alpha * unsupported_depth_edge
    )
    confidence = boundary_confidence * structure_confidence
    confidence = minimum_weight + (1.0 - minimum_weight) * confidence
    return confidence.clamp(minimum_weight, 1.0).detach()


def _masked_reliable_depth_consistency(
        reference_depth,
        warped_depth,
        reference_rgb,
        warped_rgb,
        valid_mask,
        boundary_alpha,
        structure_alpha,
        minimum_weight,
        eps=1e-6,
        reduction='mean'):
    """Scale/shift-invariant depth consistency with normalized soft weights."""
    mask = valid_mask.clamp(0.0, 1.0).detach()
    reference_normalized = _masked_standardize(reference_depth, mask, eps)
    warped_normalized = _masked_standardize(warped_depth, mask, eps)
    confidence = _depth_reliability(
        reference_depth,
        warped_depth,
        reference_rgb,
        warped_rgb,
        mask,
        boundary_alpha,
        structure_alpha,
        minimum_weight,
        eps,
    )
    weight = mask * confidence
    difference = torch.abs(reference_normalized - warped_normalized)
    count = torch.sum(weight, dim=(1, 2, 3)).clamp_min(1.0)
    loss_per_sample = torch.sum(difference * weight, dim=(1, 2, 3)) / count
    mask_count = torch.sum(mask, dim=(1, 2, 3)).clamp_min(1.0)
    confidence_per_sample = torch.sum(confidence * mask, dim=(1, 2, 3)) / mask_count
    if reduction == 'none':
        return loss_per_sample, confidence_per_sample
    if reduction != 'mean':
        raise ValueError("reduction must be 'mean' or 'none'")
    return torch.mean(loss_per_sample), torch.mean(confidence_per_sample)


def cal_reliable_true_depth_loss(
        depthInput1,
        depthInput2,
        depth_H_2to1,
        depth_H_2to1_mask,
        depth_H_1to2,
        depth_H_1to2_mask,
        depth_tps_2to1,
        depth_tps_2to1_mask,
        input1,
        input2,
        output_H,
        output_H_inv,
        warp_mesh,
        boundary_alpha=0.5,
        structure_alpha=0.5,
        minimum_weight=0.25):
    """K: spatially reliable depth guidance on the unchanged J geometry."""
    if boundary_alpha < 0.0 or structure_alpha < 0.0:
        raise ValueError('reliability alpha values must be non-negative')
    if minimum_weight < 0.0 or minimum_weight > 1.0:
        raise ValueError('minimum reliability weight must be in [0, 1]')

    h_2to1, confidence_h_2to1 = _masked_reliable_depth_consistency(
        depthInput1, depth_H_2to1, input1, output_H[:, 0:3],
        depth_H_2to1_mask, boundary_alpha, structure_alpha, minimum_weight)
    h_1to2, confidence_h_1to2 = _masked_reliable_depth_consistency(
        depthInput2, depth_H_1to2, input2, output_H_inv[:, 0:3],
        depth_H_1to2_mask, boundary_alpha, structure_alpha, minimum_weight)
    tps_2to1, confidence_tps = _masked_reliable_depth_consistency(
        depthInput1, depth_tps_2to1, input1, warp_mesh[:, 0:3],
        depth_tps_2to1_mask, boundary_alpha, structure_alpha, minimum_weight)

    loss = 3.0 * (h_2to1 + h_1to2) + tps_2to1
    confidence_mean = (
        confidence_h_2to1 + confidence_h_1to2 + confidence_tps
    ) / 3.0
    return loss, confidence_mean


def _smooth_sample_gate(value, low_threshold, full_threshold, minimum_weight):
    if full_threshold <= low_threshold:
        raise ValueError('full threshold must exceed low threshold')
    position = ((value - low_threshold) /
                (full_threshold - low_threshold)).clamp(0.0, 1.0)
    position = position * position * (3.0 - 2.0 * position)
    return minimum_weight + (1.0 - minimum_weight) * position


def _paired_information_gate(
        input1,
        input2,
        luminance_low,
        luminance_full,
        structure_low,
        structure_full,
        minimum_weight):
    """Detached pair-level support for OpenCV BGR tensors in [-1, 1].

    Unlike K's pixel reliability, this gate is deliberately not normalized
    inside a sample. It can therefore reduce the entire depth auxiliary term
    when a pair is too dark or lacks usable coarse structure. Statistics are
    measured at 64x64 to suppress sensor noise and keep training overhead low.
    """
    if input1.size(1) != 3 or input2.size(1) != 3:
        raise ValueError('information gate expects three-channel BGR inputs')
    if minimum_weight < 0.0 or minimum_weight > 1.0:
        raise ValueError('information minimum weight must be in [0, 1]')

    bgr_weights = input1.new_tensor([0.114, 0.587, 0.299]).view(1, 3, 1, 1)
    luminances = []
    structures = []
    for image in (input1, input2):
        image = ((image.detach() + 1.0) * 0.5).clamp(0.0, 1.0)
        image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
        gray = torch.sum(image * bgr_weights, dim=1, keepdim=True)
        luminances.append(gray.mean(dim=(1, 2, 3)))
        gradient_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        gradient_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        structures.append(0.5 * (
            gradient_x.mean(dim=(1, 2, 3)) + gradient_y.mean(dim=(1, 2, 3))))

    pair_luminance = 0.5 * (luminances[0] + luminances[1])
    pair_structure = 0.5 * (structures[0] + structures[1])
    luminance_gate = _smooth_sample_gate(
        pair_luminance, luminance_low, luminance_full, minimum_weight)
    structure_gate = _smooth_sample_gate(
        pair_structure, structure_low, structure_full, minimum_weight)
    combined_gate = torch.minimum(luminance_gate, structure_gate)
    return (
        combined_gate.detach(),
        luminance_gate.detach(),
        structure_gate.detach(),
        pair_luminance.detach(),
        pair_structure.detach(),
    )


def cal_information_reliable_true_depth_loss(
        depthInput1,
        depthInput2,
        depth_H_2to1,
        depth_H_2to1_mask,
        depth_H_1to2,
        depth_H_1to2_mask,
        depth_tps_2to1,
        depth_tps_2to1_mask,
        input1,
        input2,
        output_H,
        output_H_inv,
        warp_mesh,
        boundary_alpha=0.5,
        structure_alpha=0.5,
        minimum_weight=0.25,
        sample_luminance_low=0.22277420572936543,
        sample_luminance_full=0.40093238055706026,
        sample_structure_low=0.025516219437122345,
        sample_structure_full=0.05054995194077492,
        sample_minimum_weight=0.25):
    """L: K pixel reliability plus training-calibrated sample reliability."""
    h_2to1, confidence_h_2to1 = _masked_reliable_depth_consistency(
        depthInput1, depth_H_2to1, input1, output_H[:, 0:3],
        depth_H_2to1_mask, boundary_alpha, structure_alpha, minimum_weight,
        reduction='none')
    h_1to2, confidence_h_1to2 = _masked_reliable_depth_consistency(
        depthInput2, depth_H_1to2, input2, output_H_inv[:, 0:3],
        depth_H_1to2_mask, boundary_alpha, structure_alpha, minimum_weight,
        reduction='none')
    tps_2to1, confidence_tps = _masked_reliable_depth_consistency(
        depthInput1, depth_tps_2to1, input1, warp_mesh[:, 0:3],
        depth_tps_2to1_mask, boundary_alpha, structure_alpha, minimum_weight,
        reduction='none')
    (sample_gate, luminance_gate, structure_gate,
     pair_luminance, pair_structure) = _paired_information_gate(
        input1, input2, sample_luminance_low, sample_luminance_full,
        sample_structure_low, sample_structure_full, sample_minimum_weight)

    loss_per_sample = 3.0 * (h_2to1 + h_1to2) + tps_2to1
    loss = torch.mean(loss_per_sample * sample_gate)
    confidence_mean = torch.mean(
        (confidence_h_2to1 + confidence_h_1to2 + confidence_tps) / 3.0)
    return (
        loss,
        confidence_mean,
        torch.mean(sample_gate),
        torch.mean(luminance_gate),
        torch.mean(structure_gate),
        torch.mean(pair_luminance),
        torch.mean(pair_structure),
    )
