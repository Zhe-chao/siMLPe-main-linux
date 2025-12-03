import argparse
import os
import sys
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# Ensure shared lib can be imported when script executed directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LIB_PATH = os.path.join(PROJECT_ROOT, "lib")
if LIB_PATH not in sys.path:
    sys.path.insert(0, LIB_PATH)

from utils.misc import _some_variables  # noqa: E402


def build_edges() -> Iterable[Tuple[int, int]]:
    parent, _, _, _ = _some_variables()
    return [(int(child), int(parent[child])) for child in range(len(parent)) if parent[child] >= 0]


EDGES = list(build_edges())
ROOT_INDEX = 0


def project_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Project 3D joints to 2D using a simple orthographic camera."""
    centered = points - points[ROOT_INDEX]
    coords = centered[:, [0, 2]]
    coords[:, 1] *= -1
    max_range = np.max(np.linalg.norm(coords, axis=1))
    if max_range < 1e-6:
        scale = 1.0
    else:
        scale = 0.45 * min(width, height) / max_range
    coords = coords * scale
    coords += np.array([width / 2.0, height / 2.0])
    return coords.astype(np.int32)


def render_sequences(
    sequences: List[np.ndarray],
    output_path: str,
    fps: int,
    width: int,
    height: int,
    thickness: int = 3,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for sequence in sequences:
        for frame in sequence:
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
            pts2d = project_points(frame, width, height)
            for joint_idx in range(pts2d.shape[0]):
                cv2.circle(canvas, tuple(pts2d[joint_idx]), radius=4, color=(0, 0, 255), thickness=-1)
            for start, end in EDGES:
                cv2.line(canvas, tuple(pts2d[start]), tuple(pts2d[end]), color=(0, 150, 0), thickness=thickness)
            writer.write(canvas)

    writer.release()


def main(args: argparse.Namespace) -> None:
    data = np.load(args.source)
    if args.key not in data.files:
        raise ValueError(f"Key '{args.key}' not present in file. Available: {data.files}")
    sequences = data[args.key]
    extra_sequences: List[np.ndarray] = []
    for prefix in args.concatenate_keys:
        if prefix not in data.files:
            raise ValueError(f"Requested concat key {prefix} missing. Available: {data.files}")
        extra_sequences.append(data[prefix])

    if args.sequence_index < 0 or args.sequence_index >= sequences.shape[0]:
        raise IndexError(f"sequence_index {args.sequence_index} outside range [0, {sequences.shape[0] - 1}]")

    stitched: List[np.ndarray] = [sequences[args.sequence_index]]
    for seq_array in extra_sequences:
        stitched.append(seq_array[args.sequence_index])

    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    render_sequences(stitched, args.output_video, args.fps, args.width, args.height, thickness=args.thickness)
    print(f"Saved visualization to {args.output_video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize siMLPe predictions as 2D skeleton video.")
    parser.add_argument("--source", type=str, required=True, help="Path to npz file containing predictions.")
    parser.add_argument("--key", type=str, default="prediction", help="Primary dataset key to visualize.")
    parser.add_argument("--sequence-index", type=int, default=0, help="Sequence index within the file.")
    parser.add_argument("--output-video", type=str, default=os.path.join(PROJECT_ROOT, "visualizations", "skeleton.mp4"))
    parser.add_argument("--fps", type=int, default=25, help="Frames per second for the output video.")
    parser.add_argument("--width", type=int, default=640, help="Output video width.")
    parser.add_argument("--height", type=int, default=480, help="Output video height.")
    parser.add_argument("--thickness", type=int, default=3, help="Thickness of skeleton lines.")
    parser.add_argument(
        "--concatenate-keys",
        type=str,
        nargs="*",
        default=[],
        help="Additional dataset keys to append after the main sequence (e.g., input target).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
