#!/usr/bin/env python3
"""v41z-ex: three targeted fixes on frozen v41y-ex, nothing else.

1. Optical-flow crossfade of the cat layer over the last 8 frames of each
   241-cycle into the next cycle's first frame, to kill the n240/n241 and
   n481/n482 head-seam pop.
2. Retime gen2 so n432 peaks on a partially open frame (~g2 n188: tongue
   visible, fewer fangs than the hiss) and holds briefly, then flow-morphs
   onto the matching closing-side frame (g2 n222), skipping the widest
   jaws. The n138 gen1 meow is unchanged.
3. Close transient matte holes between the tail and the held plant so
   gen-clip pixels fill the dark gaps.

Snow, steam, window, lamp, title, audio, and the y-ex grade/flicker path
are untouched. Does not edit earlier builders or encodes.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41h as Hh
import build_v41y_ex as Y

VER = 'v41z-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_U = Y.SRC_U
SRC_Y = P / 'v41y-ex' / 'v41y-ex-span-890-915-lossless.mkv'
GEN1, GEN2 = Y.GEN1, Y.GEN2
FIN1 = P / 'v41y-ex' / 'mattes' / 'gen1-final.npz'
FIN2 = P / 'v41y-ex' / 'mattes' / 'gen2-final.npz'
PCM = Hh.PCM

V41U_SHA = Y.V41U_SHA
V41U_LOS_SHA = Y.V41U_LOS_SHA
V41Y_SHA = '1ab00563e69787f2b7309b40f2c74f75a76f796d692c823fc36e7feb4496e790'
V41Y_LOS_SHA = 'cc0a06ea9d75280dc04748e0ea8f0048060fd6c30d230e0b5e246a1a5111eb74'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
HEAD = (150, 230, 600, 920)  # reviewer head-seam band
SEAM_N = 8

# gen2 partial peak (tongue, 4 fangs, clearly bigger than n138, not the hiss)
GEN2_PEAK_SRC = 188.0
GEN2_HOLD_END = 196.0          # dst local; n432=191 is the start of the hold
GEN2_MORPH_END = 202.0         # dst local; optical-flow skip of the widest jaws
GEN2_CLOSE_SRC = 222.0         # matching ~60-70% opening on the way down
DST_B = 191.0

sha256 = Y.sha256
load_mp4 = Y.load_mp4
lerp_frame = Y.lerp_frame
lerp_alpha = Y.lerp_alpha
src_time_gen1 = Y.src_time
compose_one = Y.compose_one
write_qc = Y.write_qc
Xg, Yg = Y.Xg, Y.Yg


def luma(im):
    im = im.astype(np.float32)
    return 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]


def mad_rgb(a, b, box=None):
    if box is not None:
        y0, y1, x0, x1 = box
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def mad_luma(a, b, box=None):
    if box is not None:
        y0, y1, x0, x1 = box
        a, b = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    return float(np.abs(luma(a) - luma(b)).mean())


def fill_tail_leaf_gaps(stack, ksz=27, sig=1.0):
    """Close small bays/holes between tail and plant; fill with gen pixels later."""
    tlen = stack.shape[0]
    roi = (Xg > 960) & (Yg > 450)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    added = np.zeros_like(stack, np.float32)
    for i in range(tlen):
        a = stack[i]
        core = ((a > 0.32) & roi).astype(np.uint8)
        closed = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(closed)
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
        soft = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.3)
        add = np.clip(soft - a, 0, 1)
        add[~roi] = 0
        added[i] = add
        if i % 80 == 0:
            print(f'    gap-fill {i}/{tlen} add={(add > 0.2).sum()}', flush=True)
    added = Y.temporal_gauss(added, sig=sig)
    out = np.clip(stack + added, 0, 1)
    out[out < 0.03] = 0
    return out


def src_time_gen2(local):
    """Piecewise map that never samples the widest hiss frames.

    [0, 191] -> [0, 188]
    [191, 196] -> [188, 189]   (brief hold at the partial peak)
    [196, 202] is a flow-morph (handled in process), not a lerp
    [202, 240] -> [222, 240]
    """
    local = float(np.clip(local, 0, 240))
    if local <= DST_B:
        return local * (GEN2_PEAK_SRC / DST_B)
    if local <= GEN2_HOLD_END:
        return GEN2_PEAK_SRC + (local - DST_B) * ((189.0 - GEN2_PEAK_SRC) /
                                                  max(GEN2_HOLD_END - DST_B, 1e-6))
    if local <= GEN2_MORPH_END:
        # placeholder; process() replaces this with a flow morph
        u = (local - GEN2_HOLD_END) / max(GEN2_MORPH_END - GEN2_HOLD_END, 1e-6)
        return 189.0 + u * (GEN2_CLOSE_SRC - 189.0)
    return GEN2_CLOSE_SRC + (local - GEN2_MORPH_END) * (
        (240.0 - GEN2_CLOSE_SRC) / max(240.0 - GEN2_MORPH_END, 1e-6))


def in_meow_morph(local):
    return GEN2_HOLD_END < float(local) <= GEN2_MORPH_END


def meow_morph_w(local):
    return (float(local) - GEN2_HOLD_END) / max(GEN2_MORPH_END - GEN2_HOLD_END, 1e-6)


def in_seam(i):
    """Weight 1/SEAM .. 1 over the last SEAM frames of a cycle that has a successor."""
    if 241 - SEAM_N <= i <= 240 or 482 - SEAM_N <= i <= 481:
        local = i % NCYCLE
        return (local - (NCYCLE - SEAM_N) + 1) / float(SEAM_N)
    return None


def next_start_is_gen2(i):
    # after first cycle (ending n240) comes gen2; after second (n481) comes gen1
    return i <= 240


_dis = None


def dis():
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis


def _gray(im):
    return cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def _warp(im, flow, scale):
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0] * scale
    map_y = grid_y + flow[..., 1] * scale
    return cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def flow_blend(imA, imB, aA, aB, w):
    """Optical-flow morph, w=0 -> A, w=1 -> B. No dissolve ghost."""
    w = float(np.clip(w, 0, 1))
    if w <= 1e-6:
        return imA.astype(np.float32), aA.astype(np.float32)
    if w >= 1.0 - 1e-6:
        return imB.astype(np.float32), aB.astype(np.float32)
    A = imA.astype(np.float32)
    B = imB.astype(np.float32)
    fa = dis().calc(_gray(A), _gray(B), None)
    fb = dis().calc(_gray(B), _gray(A), None)
    A_fwd = _warp(A, fa, w)
    B_bwd = _warp(B, fb, 1.0 - w)
    aA_f = _warp(aA.astype(np.float32), fa, w)
    aB_b = _warp(aB.astype(np.float32), fb, 1.0 - w)
    # occlusion-aware-ish: prefer the source whose warped alpha is higher
    rgb = A_fwd * (1.0 - w) + B_bwd * w
    al = np.clip(aA_f * (1.0 - w) + aB_b * w, 0, 1)
    return np.clip(rgb, 0, 255), al


def sample_gen(use2, local, g1, g2, a1, a2):
    if use2:
        if in_meow_morph(local):
            w = meow_morph_w(local)
            return flow_blend(g2[189], g2[int(GEN2_CLOSE_SRC)],
                              a2[189], a2[int(GEN2_CLOSE_SRC)], w)
        t = src_time_gen2(local)
        return lerp_frame(g2, t), lerp_alpha(a2, t)
    t = src_time_gen1(local, Y.GEN1_PEAK, Y.DST_A)
    return lerp_frame(g1, t), lerp_alpha(a1, t)


def process(src_u, g1, g2, a1, a2, w_room, w_bulb, preview=None):
    want = set(preview) if preview else set(range(NFR))
    # precompute the two cycle-start (gen, alpha) targets
    start1 = (g1[0].astype(np.float32), a1[0].astype(np.float32))
    start2 = (g2[0].astype(np.float32), a2[0].astype(np.float32))
    out = [None] * NFR
    ph0 = Hh.phase_of(0)
    for i in range(NFR):
        if i not in want:
            continue
        use2, local = Y.cycle_of(i)
        gen, al = sample_gen(use2, local, g1, g2, a1, a2)
        w = in_seam(i)
        if w is not None:
            tgt_g, tgt_a = start2 if next_start_is_gen2(i) else start1
            gen, al = flow_blend(gen, tgt_g, al, tgt_a, w)
        ph = Hh.phase_of(i)
        out[i] = compose_one(src_u[i], gen, al, ph, ph0, w_room, w_bulb)
        if preview or i % 25 == 0 or w is not None or (use2 and 188 <= local <= 204):
            extra = f' seam_w={w:.3f}' if w is not None else ''
            print(f'  n{i} local={local} gen{"2" if use2 else "1"}{extra}', flush=True)
    return out


def seam_mad_report(out, src_y, dest):
    dest.mkdir(parents=True, exist_ok=True)
    typ_idx = list(range(10, 30)) + list(range(50, 70)) + list(range(250, 270))
    lines = ['pair  y_rgb  y_luma  z_rgb  z_luma']
    rows = {}
    pairs = [(239, 240), (240, 241), (241, 242),
             (480, 481), (481, 482), (482, 483),
             (0, 1), (10, 11)]
    for a, b in pairs:
        if out[a] is None or out[b] is None:
            continue
        yr, yl = mad_rgb(src_y[a], src_y[b], HEAD), mad_luma(src_y[a], src_y[b], HEAD)
        zr, zl = mad_rgb(out[a], out[b], HEAD), mad_luma(out[a], out[b], HEAD)
        rows[(a, b)] = (yr, yl, zr, zl)
        lines.append(f'{a}-{b}  {yr:.4f}  {yl:.4f}  {zr:.4f}  {zl:.4f}')
    # typical (non-seam) for z and y
    y_typ = np.mean([mad_rgb(src_y[i], src_y[i + 1], HEAD) for i in typ_idx])
    z_typ = np.mean([mad_rgb(out[i], out[i + 1], HEAD)
                     for i in typ_idx if out[i] is not None and out[i + 1] is not None])
    y_typ_l = np.mean([mad_luma(src_y[i], src_y[i + 1], HEAD) for i in typ_idx])
    z_typ_l = np.mean([mad_luma(out[i], out[i + 1], HEAD)
                       for i in typ_idx if out[i] is not None and out[i + 1] is not None])
    lines.append(f'typical  {y_typ:.4f}  {y_typ_l:.4f}  {z_typ:.4f}  {z_typ_l:.4f}')
    path = dest / 'seam-mad.txt'
    path.write_text('\n'.join(lines) + '\n')
    print('SEAM MAD\n' + '\n'.join(lines), flush=True)
    return rows, (y_typ, y_typ_l, z_typ, z_typ_l)


def z_extra_qc(out, src_y, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # y vs z meow at n432, plus u for context
    if out[432] is not None:
        def mouth(im):
            return cv2.resize(im[180:460, 520:900], None, fx=3, fy=3,
                              interpolation=cv2.INTER_NEAREST)
        pair = np.hstack([mouth(src_y[432]), mouth(out[432])])
        cv2.putText(pair, 'v41y-ex n432', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.putText(pair, 'v41z-ex n432', (mouth(src_y[432]).shape[1] + 8, 28),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.imwrite(str(dest / 'meow-n432-y-vs-z.png'), pair)
        print('meow ba', dest / 'meow-n432-y-vs-z.png')
        # also a short 3x sequence around the new peak
        cells = []
        for i in range(420, 445, 2):
            if out[i] is None:
                continue
            c = cv2.resize(out[i][180:460, 520:900], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
            cv2.putText(c, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
            cells.append(c)
        if cells:
            cv2.imwrite(str(dest / 'meow-seq-432-z.png'), np.hstack(cells))
    # seam face row y vs z
    for a, b, name in ((240, 241, '240-241'), (481, 482, '481-482')):
        if out[a] is None or out[b] is None:
            continue
        def band(im):
            return cv2.resize(im[140:280, 560:940], None, fx=2, fy=2,
                              interpolation=cv2.INTER_NEAREST)
        row = np.hstack([band(src_y[a]), band(src_y[b]), band(out[a]), band(out[b])])
        cv2.putText(row, f'y{a}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, f'y{b}', (band(src_y[a]).shape[1] + 8, 24),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, f'z{a}', (2 * band(src_y[a]).shape[1] + 8, 24),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, f'z{b}', (3 * band(src_y[a]).shape[1] + 8, 24),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.imwrite(str(dest / f'seam-y-vs-z-{name}.png'), row)
        d = np.abs(out[a].astype(np.int16) - out[b].astype(np.int16)).astype(np.uint8)
        cv2.imwrite(str(dest / f'diff-z-{name}.png'),
                    cv2.resize(np.clip(d[140:280, 560:940] * 3, 0, 255),
                               None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    # tail hole closeups y vs z
    cells = []
    for i in (108, 116, 120, 124, 128, 132, 136, 180):
        if out[i] is None:
            continue
        a = src_y[i][500:720, 980:1280]
        b = out[i][500:720, 980:1280]
        pair = np.vstack([a, b])
        cv2.putText(pair, f'y n{i}', (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
        cv2.putText(pair, 'z', (6, a.shape[0] + 20), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 2)
        cells.append(pair)
    if cells:
        cv2.imwrite(str(dest / 'tail-holes-y-vs-z.png'), np.hstack(cells))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41u-ex-mp4', P / 'v41u-ex-test-25s.mp4', V41U_SHA),
        ('v41u-ex-los', SRC_U, V41U_LOS_SHA),
        ('v41y-ex-mp4', P / 'v41y-ex-test-25s.mp4', V41Y_SHA),
        ('v41y-ex-los', SRC_Y, V41Y_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading plates, y-ex mattes, generated clips…', flush=True)
    src_u = Hh.load_span(SRC_U)
    src_y = Hh.load_span(SRC_Y)
    g1 = load_mp4(GEN1)
    g2 = load_mp4(GEN2)
    assert len(src_u) == NFR and len(src_y) == NFR
    assert len(g1) == NCYCLE and len(g2) == NCYCLE
    a1 = np.load(FIN1)['alpha'].astype(np.float32) / 255.0
    a2 = np.load(FIN2)['alpha'].astype(np.float32) / 255.0

    print('filling tail/leaf matte gaps…', flush=True)
    a1 = fill_tail_leaf_gaps(a1)
    a2 = fill_tail_leaf_gaps(a2)
    WORK.mkdir(exist_ok=True)
    (WORK / 'mattes').mkdir(exist_ok=True)
    np.savez_compressed(WORK / 'mattes' / 'gen1-final.npz', alpha=(a1 * 255).astype(np.uint8))
    np.savez_compressed(WORK / 'mattes' / 'gen2-final.npz', alpha=(a2 * 255).astype(np.uint8))

    pane_m = __import__('build_v41t_ex', fromlist=['T']).pane_masks().astype(np.float32)
    import build_v41t_ex as T
    cat_union = np.clip(np.maximum(a1[0], a2[0]), 0, 1)
    w_room, w_bulb = T.lamp_weights(pane_m, cat_union)
    QC.mkdir(parents=True, exist_ok=True)

    if args.preview:
        idx = sorted(set(
            list(range(0, 241, 10))
            + list(range(108, 141))
            + list(range(126, 147))
            + list(range(232, 252))
            + list(range(420, 445))
            + list(range(473, 500))
            + [599]
        ))
        print('preview frames', len(idx), flush=True)
        out = process(src_u, g1, g2, a1, a2, w_room, w_bulb, preview=idx)
        dest = QC / 'preview'
        Y.VER = VER
        write_qc(out, src_u, dest, mp4=None)
        z_extra_qc(out, src_y, src_u, dest)
        seam_mad_report(out, src_y, dest)
        print('preview written', dest, flush=True)
        return

    out = process(src_u, g1, g2, a1, a2, w_room, w_bulb)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_u, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical',
          len(bad) if bad else '', flush=True)

    los = WORK / f'{VER}-span-890-915-lossless.mkv'
    print('writing lossless…', flush=True)
    Hh.write_lossless(out, los)
    print('encoding mp4…', flush=True)
    Hh.encode_mp4(out, mp4)
    print(VER, 'mp4', sha256(mp4), flush=True)
    print(VER, 'lossless', sha256(los), flush=True)
    shutil.copy2(mp4, WORK / f'{VER}-test-25s.mp4')
    shutil.copy2(Path(__file__), WORK / 'build_v41z_ex.py')
    dest = QC / 'encoded'
    Y.VER = VER  # labels in shared QC helpers
    write_qc(out, src_u, dest, mp4=mp4)
    z_extra_qc(out, src_y, src_u, dest)
    seam_mad_report(out, src_y, dest)


if __name__ == '__main__':
    main()
