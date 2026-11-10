#!/usr/bin/env python3
"""v41u-ex: redo the r-ex mouth without the chest leak, then t-ex passes.

v41r-ex restore_mouth used a loose 'mauve' test (Y<135, R>G+5, B>G*0.65)
on a box that includes the right cheek/chest (x610-830, y345-475). Dusk
cream fur passes that test, so a big unfeathered region of colour-matched
daylight v41 was composited onto the chest — a pale rectangle with orange
marks. Inherited by s-ex and t-ex.

This builder starts from frozen v41q-ex lossless, reapplies r-ex dust and
ears, uses a tight mouth mask (mauve only next to the opening; mask clipped
to the mouth, never the right chest), then reapplies the v41t-ex frost
left-bank plate, lamp integration, tail chain, and n432 muzzle patch.

Does not edit or overwrite earlier named encodes or builders.
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
import build_v41r_ex as R
import build_v41t_ex as T

VER = 'v41u-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_Q = P / 'v41q-ex' / 'v41q-ex-span-890-915-lossless.mkv'
SRC_V = P / 'work' / 'v41-span-890-915-lossless.mkv'
MATTE = Hh.MATTE
PCM = Hh.PCM

V41_SHA = Hh.V41_SHA
V41_LOS_SHA = '226206bf77a1e30bb4e9daf44d9885897edf3079f1043376c5d109bc4025f796'
V41Q_SHA = 'be65528916d68b99cc64546d865388f824656b3975adb8f861f10d03e548afcd'
V41Q_LOS_SHA = '9e62b006a28977f54ac5f0a0db9e25b118c1837922cd6796e9d032ad0b0dfe6e'
V41R_SHA = 'b216388508c7b48e12866f135ba30971097dd3af10f81933e95c8dd9baaf8e1a'
V41T_SHA = 'aaeb5dbeba9360ec0118e9a9bc5d4d116e1e087a99b63b907ec0ca98bc064c07'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def restore_mouth_tight(img, donor):
    """Same v41 mouth grade as r-ex, but the mask cannot reach the chest.

    Mauve is only kept next to a detected opening, and is clipped well
    before the right cheek (the r-ex test used xx>200 on a 220-wide box,
    which still included x 750-830). Opening detections on the far right
    or low chest are dropped.
    """
    imgf = img.astype(np.float32)
    don = donor.astype(np.float32)
    y0, y1, x0, x1 = 345, 475, 610, 830
    sub_q = imgf[y0:y1, x0:x1]
    sub_v = don[y0:y1, x0:x1]
    Yv = T.luma(sub_v)
    B, G, Rc = sub_v[..., 0], sub_v[..., 1], sub_v[..., 2]
    opening = ((Yv < 110) & (Rc > G + 8) & (Rc > B + 5) & (Yv > 22)).astype(np.uint8)
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
    opening[xx > 165] = 0  # abs x > 775
    opening[yy > 85] = 0   # abs y > 430
    Yq = T.luma(sub_q)
    Bq, Gq, Rq = sub_q[..., 0], sub_q[..., 1], sub_q[..., 2]
    mauve = ((Rq > Gq + 5) & (Bq > Gq * 0.65) & (Yq > 24) & (Yq < 135)
             & (Rq < 190) & (Rq > 50)).astype(np.uint8)
    mauve[yy < 20] = 0
    mauve[yy > 90] = 0
    mauve[xx < 15] = 0
    mauve[xx > 160] = 0  # abs x > 770; r-ex used 200 and leaked onto the chest
    ker11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ker19 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    if opening.any():
        mauve = mauve * cv2.dilate(opening, ker19)
        ys, xs = np.where(opening > 0)
        cx, cy = float(np.median(xs)), float(np.median(ys))
        chin_cy = float(ys.max()) + 12.0
        width = max(22.0, 0.48 * (xs.max() - xs.min() + 6))
        around = cv2.dilate(np.maximum(opening, mauve), ker19).astype(np.float32)
        x_right = float(xs.max()) + 14.0
        y_bot = float(ys.max()) + 22.0
    elif mauve.any():
        ys, xs = np.where(mauve > 0)
        cx, cy = float(np.median(xs)), float(np.median(ys))
        chin_cy = float(ys.max()) + 10.0
        width = max(22.0, 0.48 * (xs.max() - xs.min() + 6))
        around = cv2.dilate(mauve, ker19).astype(np.float32)
        x_right = float(xs.max()) + 14.0
        y_bot = float(ys.max()) + 22.0
    else:
        nose = ((Rc > 80) & (Rc > G + 10) & (Rc > B + 20) & (Yv > 50) & (Yv < 170) & (yy < 50))
        if nose.any():
            ys, xs = np.where(nose)
            cx, cy = float(np.median(xs)), float(ys.max()) + 16.0
        else:
            cx, cy = 100.0, 38.0
        chin_cy = cy + 18.0
        width = 26.0
        around = np.zeros_like(yy, np.float32)
        x_right = cx + 55.0
        y_bot = chin_cy + 20.0
    mouth_slot = np.exp(-0.5 * (((xx - cx) / (width * 0.80)) ** 2 + ((yy - cy) / 14.0) ** 2))
    chin = np.exp(-0.5 * (((xx - cx) / (width * 0.90)) ** 2 + ((yy - chin_cy) / 16.0) ** 2))
    m = np.clip(np.maximum.reduce([around, mouth_slot, 0.80 * chin]), 0, 1)
    m = m * T.sstep((x_right - xx) / 8.0)
    m = m * T.sstep((y_bot - yy) / 8.0)
    m = m * T.sstep((155.0 - (xx - cx)) / 8.0)
    m = cv2.GaussianBlur(m, (0, 0), 1.6)
    feat = (T.sstep(xx / 12.0) * T.sstep((x1 - x0 - 1 - xx) / 12.0)
            * T.sstep(yy / 10.0) * T.sstep((y1 - y0 - 1 - yy) / 12.0))
    m = np.clip(m * feat, 0, 1)
    fur_s = (opening == 0) & (mauve == 0) & (m > 0.05) & (Yq > 30) & (Yv > 50)
    ga, gb = R.affine_match(sub_v, sub_q, fur_s)
    graded = R.apply_ab(sub_v, ga, gb)
    if opening.any() or mauve.any():
        op = np.maximum(opening, cv2.erode(mauve, ker11)).astype(np.float32)
        op = cv2.GaussianBlur(op, (0, 0), 0.7)
        g = graded.copy()
        excess_b = np.clip(g[..., 0] - 0.62 * g[..., 1], 0, None)
        g[..., 0] = np.clip(g[..., 0] - 1.00 * excess_b, 0, 255)
        g[..., 1] = np.clip(g[..., 1] * 1.04 + 3.0, 0, 255)
        g[..., 2] = np.clip(g[..., 2] * 1.10 + 8.0, 0, 255)
        graded = graded * (1.0 - 0.90 * op[..., None]) + g * (0.90 * op[..., None])
    sub = sub_q * (1.0 - m[..., None]) + graded * m[..., None]
    out = imgf.copy()
    out[y0:y1, x0:x1] = sub
    return np.clip(out, 0, 255)


def process_one(src_q, src_v, cat_a, ph, ctx, i):
    img = R.remove_dust(src_q, cat_a)
    img = R.restore_ears(img, src_v, cat_a)
    img = restore_mouth_tight(img, src_v)
    if 418 <= i <= 448:
        img = T.fix_muzzle(img)
    img, _ = T.apply_tail(img, cat_a, ph, ctx['tail'], ctx['mic'], ctx['tail_bg'], ctx['occ'])
    img = T.apply_plant_flicks(img, ctx['plate'], cat_a, ph, ctx['pdelay'], ctx['rest_plant'])
    img = T.apply_frost(img, ctx['frost_plate'], ctx['frost_m'], ctx['pane'])
    img = T.apply_flicker(img, ctx['w_room'], ctx['w_bulb'], ph)
    img = np.clip(img, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = TITLE
    img = img.copy()
    img[y0:y1, x0:x1] = src_q[y0:y1, x0:x1]
    return img


def process(src_q, src_v, mattes, ctx, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = Hh.phase_of(i)
        out[i] = process_one(src_q[i], src_v[i], T.norm_alpha(mattes[i]), ph, ctx, i)
        if preview or i % 25 == 0:
            print(f'  n{i} ph={ph}', flush=True)
    return out


def chest_grid(out, src_q, dest, src_r=None):
    """Chest/cheek crop every 10 frames over one cycle, vs q-ex."""
    dest.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = 300, 520, 680, 900
    tiles_u, tiles_q, tiles_d, tiles_r = [], [], [], []
    report = []
    for i in range(0, 241, 10):
        if out[i] is None:
            continue
        u = out[i][y0:y1, x0:x1]
        q = src_q[i][y0:y1, x0:x1]
        mad = float(np.mean(np.abs(u.astype(np.float32) - q.astype(np.float32))))
        mx = float(np.max(np.abs(u.astype(np.int16) - q.astype(np.int16))))
        report.append((i, mad, mx))
        def lab(im, tag):
            z = cv2.resize(im, (200, 200), interpolation=cv2.INTER_AREA)
            cv2.putText(z, tag, (6, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            return z
        tiles_u.append(lab(u, f'u n{i}'))
        tiles_q.append(lab(q, f'q n{i}'))
        diff = np.clip(np.abs(u.astype(np.float32) - q.astype(np.float32)) * 6.0, 0, 255)
        tiles_d.append(lab(diff.astype(np.uint8), f'd n{i}'))
        cv2.imwrite(str(dest / f'chest-u-2x-n{i}.png'),
                    cv2.resize(u, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        if src_r is not None and src_r[i] is not None:
            tiles_r.append(lab(src_r[i][y0:y1, x0:x1], f'r n{i}'))
    def sheet(cells, name, cols=5):
        if not cells:
            return
        cells = list(cells)
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
        path = dest / name
        cv2.imwrite(str(path), np.vstack(rows))
        print('chest grid', path)
    sheet(tiles_u, 'chest-grid-u.png')
    sheet(tiles_q, 'chest-grid-q.png')
    sheet(tiles_d, 'chest-grid-diff-vs-q.png')
    if tiles_r:
        sheet(tiles_r, 'chest-grid-r.png')
    print('chest vs q-ex every 10 frames:')
    for i, mad, mx in report:
        print(f'  n{i:3d} mad={mad:.2f} max={mx:.1f}')
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41', Hh.V41_MP4, V41_SHA),
        ('v41-los', SRC_V, V41_LOS_SHA),
        ('v41q-ex-mp4', P / 'v41q-ex-test-25s.mp4', V41Q_SHA),
        ('v41q-ex-los', SRC_Q, V41Q_LOS_SHA),
        ('v41r-ex-mp4', P / 'v41r-ex-test-25s.mp4', V41R_SHA),
        ('v41t-ex-mp4', P / 'v41t-ex-test-25s.mp4', V41T_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading q-ex / v41 lossless…', flush=True)
    src_q = Hh.load_span(SRC_Q)
    src_v = Hh.load_span(SRC_V)
    assert len(src_q) == NFR and len(src_v) == NFR
    print('loading cat mattes…', flush=True)
    mattes = Hh.load_span(MATTE, gray=True)

    print('preparing tail / frost / lamp (t-ex passes)…', flush=True)
    tail = T.prep_tail_padded(T.SPRITE_FIXED)
    mic = Hh._mic_mask()
    # tail/plant plates from q-ex (same tail region as r-ex)
    tail_bg, occ = T.build_tail_bg(src_q, mattes, tail)
    plate = T.G.build_plant_plate(src_q[121], T.norm_alpha(mattes[121]))
    pdelay = T.plant_delay_map(plate)
    rest_plant = src_q[121].astype(np.float32)
    pane = T.pane_masks()
    cat_union = np.zeros((H, W), np.float32)
    for i in range(0, 241, 3):
        cat_union = np.maximum(cat_union, T.norm_alpha(mattes[i]))
    cat_union = cv2.GaussianBlur(cat_union, (0, 0), 4.0)
    frost_plate, frost_m = T.build_frost(src_q, cat_union)
    w_room, w_bulb = T.lamp_weights(pane, cat_union)
    ctx = dict(tail=tail, mic=mic, tail_bg=tail_bg, occ=occ,
               plate=plate, pdelay=pdelay, rest_plant=rest_plant,
               frost_plate=frost_plate, frost_m=frost_m, pane=pane,
               w_room=w_room, w_bulb=w_bulb)
    QC.mkdir(parents=True, exist_ok=True)
    print(f'frost on face crop: mean={frost_m[160:460,570:950].mean():.4f} '
          f'max={frost_m[160:460,570:950].max():.4f}', flush=True)

    # optional r-ex frames for the chest comparison grid (preview/encode QC)
    src_r = None

    if args.preview:
        idx = sorted(set(list(range(0, 241)) + [241, 300, 360, 370, 380, 432, 450]))
        out = process(src_q, src_v, mattes, ctx, preview=idx)
        dest = QC / 'preview'
        dest.mkdir(parents=True, exist_ok=True)
        T.tail_strip(out, dest)
        T.face_compare_grid(out, src_q, dest)  # vs q-ex; lamp-adjusted
        T.mouth_qc(out, src_q, dest)
        chest_grid(out, src_q, dest)
        # also write key 2x heads and chests
        for i in (100, 130, 138, 150, 370, 432):
            if out[i] is None:
                continue
            cv2.imwrite(str(dest / f'head-2x-n{i}.jpg'),
                        cv2.resize(out[i][160:460, 570:950], None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            cv2.imwrite(str(dest / f'chest-2x-n{i}.png'),
                        cv2.resize(out[i][340:500, 720:920], None, fx=2, fy=2,
                                   interpolation=cv2.INTER_NEAREST))
        WORK.mkdir(exist_ok=True)
        T.tail_strip(out, WORK / 'preview')
        chest_grid(out, src_q, WORK / 'preview')
        print('preview written', dest, flush=True)
        return

    out = process(src_q, src_v, mattes, ctx)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_q, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical',
          len(bad) if bad else '', flush=True)

    WORK.mkdir(exist_ok=True)
    los = WORK / f'{VER}-span-890-915-lossless.mkv'
    print('writing lossless…', flush=True)
    Hh.write_lossless(out, los)
    print('encoding mp4…', flush=True)
    Hh.encode_mp4(out, mp4)
    print(VER, 'mp4', sha256(mp4), flush=True)
    print(VER, 'lossless', sha256(los), flush=True)
    shutil.copy2(mp4, WORK / f'{VER}-test-25s.mp4')
    shutil.copy2(Path(__file__), WORK / 'build_v41u_ex.py')
    print('v41 still', sha256(Hh.V41_MP4), flush=True)
    print('v41q-ex still', sha256(P / 'v41q-ex-test-25s.mp4'), flush=True)
    print('v41r-ex still', sha256(P / 'v41r-ex-test-25s.mp4'), flush=True)
    print('v41t-ex still', sha256(P / 'v41t-ex-test-25s.mp4'), flush=True)
    (QC / 'encoded-tmp').mkdir(parents=True, exist_ok=True)
    T.plot_lamp(out, src_q, QC / 'encoded-tmp')
    T.tail_strip(out, QC / 'encoded-tmp')
    T.face_compare_grid(out, src_q, QC / 'encoded-tmp')
    T.mouth_qc(out, src_q, QC / 'encoded-tmp')
    chest_grid(out, src_q, QC / 'encoded-tmp')


if __name__ == '__main__':
    main()
