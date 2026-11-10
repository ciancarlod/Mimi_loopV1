#!/usr/bin/env python3
"""Render a few v41ag-ex test frames from the small CI asset pack.

Does not call the original builder's main(): it skips frozen-SHA checks,
loads sliced stacks from the pack, and writes PNGs. Builder sources are
imported from --builders-dir after rewriting hard-coded /workspace paths
to MIMI_ROOT so a runner never needs the full project tree.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


BUILDER_FILES = [
    'build_v41ag_ex.py',
    'build_v41h.py',
    'build_v41y_ex.py',
    'build_v41z_ex.py',
    'build_v41t_ex.py',
    'build_v41aa_ex.py',
    'build_v41c.py',
    'build_v41f_r2.py',
    'build_v41g.py',
]


class SparseStack:
    """241-or-N length sequence that only holds a few frames."""

    def __init__(self, indices, frames, length):
        self._d = {int(i): frames[k] for k, i in enumerate(indices)}
        self._n = int(length)

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise TypeError('slice not supported on SparseStack')
        i = int(i)
        if i < 0:
            i += self._n
        try:
            return self._d[i]
        except KeyError as e:
            raise IndexError(f'sparse stack missing frame {i}') from e


def parse_args():
    here = Path(__file__).resolve()
    overlay = here.parent.parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames', default='0,42,138,432',
                    help='comma-separated output frame indices')
    ap.add_argument('--out', required=True, help='directory for PNG outputs')
    ap.add_argument('--root', default=os.environ.get('MIMI_ROOT'),
                    help='extracted pack root (or set MIMI_ROOT)')
    ap.add_argument('--builders-dir',
                    default=os.environ.get('MIMI_BUILDERS') or str(overlay / 'builders'),
                    help='directory containing build_v41*.py')
    return ap.parse_args()


def import_builders(builders_dir: Path, root: Path, lamp: Path):
    """Load builder modules from rewritten copies so P and the lamp path
    point at the pack, not /workspace/lofi-v41-meow."""
    builders_dir = builders_dir.resolve()
    missing = [n for n in BUILDER_FILES if not (builders_dir / n).is_file()]
    if missing:
        raise SystemExit(f'builders not found in {builders_dir}: {missing}')

    work = Path(tempfile.mkdtemp(prefix='mimi-builders-'))
    root_lit = repr(str(root))
    lamp_lit = repr(str(lamp))
    for name in BUILDER_FILES:
        text = (builders_dir / name).read_text()
        text = text.replace("Path('/workspace/lofi-v41-meow')", f'Path({root_lit})')
        text = text.replace("'/workspace/lofi-v28/lamp-mask.png'", lamp_lit)
        # Keep unused v24/v30 constants from pointing at the real box paths.
        text = text.replace("Path('/workspace/lofi-v24/v24-original-mimi-one-cycle-mild-squint.mp4')",
                            f"Path({root_lit}) / 'unused-v24.mp4'")
        text = text.replace("'/workspace/lofi-v30/safe-window-mask.png'",
                            repr(str(root / 'unused-window-mask.png')))
        (work / name).write_text(text)

    # Prefer the rewritten copies over anything else on sys.path.
    sys.path.insert(0, str(work))
    import build_v41ag_ex as ag  # noqa: E402
    return ag, work


def load_layer_pack(path: Path, length: int = 241):
    z = np.load(path)
    idx = [int(i) for i in z['indices']]
    return (
        SparseStack(idx, z['gen'], length),
        SparseStack(idx, z['cat'], length),
        SparseStack(idx, z['plant'], length),
        SparseStack(idx, z['chair'], length),
        idx,
    )


def load_plates(path: Path, length: int = 600) -> list:
    z = np.load(path)
    idx = [int(i) for i in z['indices']]
    frames = z['frames']
    src_u = [None] * length
    for k, i in enumerate(idx):
        src_u[i] = frames[k]
    return src_u


def install_plant_matte(ag, filled_u8: np.ndarray):
    """Rebuild the exact u_plant_matte result from the packed silhouette."""
    # packed PNG is 0/255; accept 0/1 too
    filled = (filled_u8 > 127).astype(np.uint8) if filled_u8.max() > 1 else filled_u8.astype(np.uint8)
    a = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.6)
    ag._U_PLANT = np.clip(a, 0, 1)
    return ag._U_PLANT


def main():
    args = parse_args()
    if not args.root:
        raise SystemExit('set MIMI_ROOT or pass --root to the extracted pack')
    root = Path(args.root).resolve()
    builders_dir = Path(args.builders_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted({int(x) for x in args.frames.split(',') if x.strip() != ''})
    lamp = root / 'vendor' / 'lofi-v28' / 'lamp-mask.png'
    if not lamp.is_file():
        raise SystemExit(f'missing lamp mask {lamp}')
    blink = root / 'qc' / 'v41ad-ex' / 'v41ad-blink-i2v.mp4'
    if not blink.is_file():
        print(f'warning: blink clip missing {blink}', flush=True)

    ag, work = import_builders(builders_dir, root, lamp)
    try:
        import build_v41t_ex as T

        print('loading sliced layers from', root, flush=True)
        gi1, c1, p1, h1, _ = load_layer_pack(root / 'g1-layers.npz')
        gi2, c2, p2, h2, _ = load_layer_pack(root / 'g2-layers.npz')
        src_u = load_plates(root / 'u-ex-plates.npz')

        filled = cv2.imread(str(root / 'u-plant-filled.png'), cv2.IMREAD_GRAYSCALE)
        if filled is None:
            raise SystemExit(f'cannot read {root / "u-plant-filled.png"}')
        install_plant_matte(ag, filled)

        # Cat slices are already repaired; do not call repair_cat_stack
        # (the full ear-prior donor frames are not in the pack).
        cat_union = np.clip(
            np.maximum(c1[0].astype(np.float32), c2[0].astype(np.float32)) / 255.0,
            0, 1,
        )
        pane = T.pane_masks().astype(np.float32)
        w_room, w_bulb = T.lamp_weights(pane, cat_union)

        print('rendering frames', frames, flush=True)
        out = ag.process(src_u, c1, c2, p1, p2, h1, h2, gi1, gi2,
                         w_room, w_bulb, preview=frames)
        written = []
        for i in frames:
            im = out[i]
            if im is None:
                raise SystemExit(f'process() returned no frame n{i}')
            dest = out_dir / f'n{i:03d}.png'
            if not cv2.imwrite(str(dest), im, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise SystemExit(f'failed to write {dest}')
            written.append(dest)
            print(f'wrote {dest} shape={im.shape}', flush=True)
        print('done', len(written), 'frames', flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
