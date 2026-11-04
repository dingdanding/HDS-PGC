"""Regression, calibration and backward checks for rebuilt DSFN-L."""
import importlib.util
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "Warp" / "newCodes"))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


l_loss = load("l_loss_candidate", HERE / "Warp" / "newCodes" / "loss.py")


def tensors(batch=3, height=16, width=16):
    generator = torch.Generator().manual_seed(2026)
    depth1 = torch.rand(batch, 1, height, width, generator=generator) * 255.0
    depth2 = torch.rand(batch, 1, height, width, generator=generator) * 255.0
    warped1 = (depth1 * 0.9 + 4.0).clone().requires_grad_(True)
    warped2 = (depth2 * 1.1 - 3.0).clone().requires_grad_(True)
    warped_tps = (depth1 * 0.95 + 2.0).clone().requires_grad_(True)
    mask = torch.ones(batch, 1, height, width)
    input1 = torch.rand(batch, 3, height, width, generator=generator) * 2.0 - 1.0
    input2 = torch.rand(batch, 3, height, width, generator=generator) * 2.0 - 1.0
    output_h = torch.cat((input1, mask), dim=1)
    output_h_inv = torch.cat((input2, mask), dim=1)
    warp_mesh = torch.cat((input1, mask), dim=1)
    return (depth1, depth2, warped1, mask, warped2, mask, warped_tps, mask,
            input1, input2, output_h, output_h_inv, warp_mesh)


def main():
    args = tensors()
    old_value, old_confidence = l_loss.cal_reliable_true_depth_loss(*args)
    full_gate_result = l_loss.cal_information_reliable_true_depth_loss(
        *args,
        sample_luminance_low=-1.0,
        sample_luminance_full=-0.5,
        sample_structure_low=-1.0,
        sample_structure_full=-0.5,
    )
    torch.testing.assert_close(full_gate_result[0], old_value, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full_gate_result[1], old_confidence, rtol=1e-6, atol=1e-6)

    dark = torch.full((2, 3, 64, 64), -1.0)
    flat_normal = torch.zeros((2, 3, 64, 64))
    structured = torch.zeros((2, 3, 64, 64))
    structured[:, :, :, ::2] = 1.0
    dark_result = l_loss._paired_information_gate(
        dark, dark, 0.22277420572936543, 0.40093238055706026,
        0.025516219437122345, 0.05054995194077492, 0.25)
    flat_result = l_loss._paired_information_gate(
        flat_normal, flat_normal, 0.22277420572936543, 0.40093238055706026,
        0.025516219437122345, 0.05054995194077492, 0.25)
    structured_result = l_loss._paired_information_gate(
        structured, structured, 0.22277420572936543, 0.40093238055706026,
        0.025516219437122345, 0.05054995194077492, 0.25)
    torch.testing.assert_close(dark_result[0], torch.full_like(dark_result[0], 0.25))
    torch.testing.assert_close(flat_result[0], torch.full_like(flat_result[0], 0.25))
    torch.testing.assert_close(structured_result[0], torch.ones_like(structured_result[0]))

    outputs = l_loss.cal_information_reliable_true_depth_loss(*args)
    if not all(torch.isfinite(item).all() for item in outputs):
        raise RuntimeError("L produced a non-finite value")
    outputs[0].backward()
    for tensor in (args[2], args[4], args[6]):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise RuntimeError("L backward failed")
    if not 0.25 <= outputs[2].item() <= 1.0:
        raise RuntimeError("sample information gate escaped configured bounds")
    print("PASS: unchanged K loss is numerically identical")
    print("PASS: dark and flat pairs attenuated; structured normal pair preserved")
    print("PASS: rebuilt L forward/backward finite")
    print("sample_gate={:.6f} luminance_gate={:.6f} structure_gate={:.6f}".format(
        outputs[2].item(), outputs[3].item(), outputs[4].item()))


if __name__ == "__main__":
    main()
