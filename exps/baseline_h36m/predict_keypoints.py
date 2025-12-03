import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project libs are resolvable before importing.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
LIB_PATH = os.path.join(PROJECT_ROOT, 'lib')
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

from config import config  # noqa: E402
from datasets.h36m_eval import H36MEval  # noqa: E402
from model import siMLPe as Model  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load checkpoint while handling common key prefix mismatches."""
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("motion_transformer.transformer."):
            new_key = new_key.replace("motion_transformer.transformer", "motion_mlp.mlps", 1)
        remapped[new_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint ignored: {unexpected}")
    if missing:
        raise RuntimeError(f"Missing parameters after remapping: {missing}")


def get_dct_matrix(length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create DCT / IDCT matrices matching the evaluation setup."""
    dct_m = np.eye(length)
    for k in np.arange(length):
        for i in np.arange(length):
            w = np.sqrt(2 / length)
            if k == 0:
                w = np.sqrt(1 / length)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / length)
    idct_m = np.linalg.inv(dct_m)
    dct_m = torch.tensor(dct_m, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    idct_m = torch.tensor(idct_m, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    return dct_m, idct_m


def autoregressive_predict(
    model: torch.nn.Module,
    motion_input: torch.Tensor,
    dct_m: torch.Tensor,
    idct_m: torch.Tensor,
    eval_cfg,
) -> torch.Tensor:
    """Generate future poses autoregressively, mirroring baseline evaluation."""
    joint_used_xyz = np.array(
        [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 21, 22, 25, 26, 27, 29, 30],
        dtype=np.int64,
    )
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31], dtype=np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30], dtype=np.int64)

    b, n, c, _ = motion_input.shape
    motion_input = motion_input.reshape(b, n, 32, 3)
    motion_input = motion_input[:, :, joint_used_xyz].reshape(b, n, -1)

    outputs: List[torch.Tensor] = []
    step = config.motion.h36m_target_length_train
    num_step = 1 if step == eval_cfg.motion.h36m_target_length else int(np.ceil(eval_cfg.motion.h36m_target_length / step))

    for _ in range(num_step):
        with torch.no_grad():
            if config.deriv_input:
                motion_input_ = torch.matmul(dct_m[:, :, :config.motion.h36m_input_length], motion_input.to(DEVICE))
            else:
                motion_input_ = motion_input.to(DEVICE)
            output = model(motion_input_)
            output = torch.matmul(idct_m[:, :config.motion.h36m_input_length, :], output)[:, :step, :]
            if config.deriv_output:
                output = output + motion_input[:, -1:, :].to(DEVICE).repeat(1, step, 1)
        outputs.append(output.cpu())
        motion_input = torch.cat([motion_input[:, step:], output.cpu()], dim=1)

    motion_pred = torch.cat(outputs, dim=1)[:, : eval_cfg.motion.h36m_target_length]

    motion_pred = motion_pred.reshape(b, -1, 22, 3)
    full_seq = torch.zeros(b, motion_pred.size(1), 32, 3, dtype=motion_pred.dtype)
    full_seq[:, :, joint_used_xyz, :] = motion_pred

    # Copy joints that were ignored to match the original 32-joint format.
    full_seq[:, :, joint_to_ignore, :] = full_seq[:, :, joint_equal, :]
    return full_seq


def run_inference(args: argparse.Namespace) -> None:
    """Entry point for running inference and exporting pose sequences."""
    model = Model(config)
    load_checkpoint(model, args.model_pth)
    model.to(DEVICE)
    model.eval()

    eval_cfg = copy.deepcopy(config)
    eval_cfg.motion.h36m_target_length = config.motion.h36m_target_length_eval

    dataset = H36MEval(eval_cfg, "test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    dct_m, idct_m = get_dct_matrix(config.motion.h36m_input_length_dct)
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    inputs: List[np.ndarray] = []

    progress = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, (motion_input, motion_target) in progress:
        if args.max_batches and batch_idx >= args.max_batches:
            break
        pred = autoregressive_predict(model, motion_input, dct_m, idct_m, eval_cfg)
        predictions.append(pred.numpy())  # meters, shape: B x T x 32 x 3
        targets.append(motion_target.numpy())
        inputs.append(motion_input.numpy())

    if not predictions:
        raise RuntimeError("No predictions generated. Check dataset or arguments.")

    predictions_np = np.concatenate(predictions, axis=0)
    targets_np = np.concatenate(targets, axis=0)
    inputs_np = np.concatenate(inputs, axis=0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.savez_compressed(
        args.output_path,
        prediction=predictions_np,
        target=targets_np,
        input=inputs_np,
    )

    metadata: Dict[str, int] = {
        "num_sequences": predictions_np.shape[0],
        "prediction_timesteps": predictions_np.shape[1],
        "num_joints": predictions_np.shape[2],
        "input_timesteps": inputs_np.shape[1],
    }
    with open(args.output_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved predictions to {args.output_path}")
    print(f"Metadata: {metadata}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export siMLPe future pose predictions.")
    parser.add_argument("--model-pth", type=str, required=True, help="Path to trained siMLPe checkpoint.")
    parser.add_argument(
        "--output-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "h36m_predictions.npz"),
        help="npz file to store predicted and target poses (meters).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional limit on processed batches (0 = all).")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
