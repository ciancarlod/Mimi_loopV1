#!/usr/bin/env python3
"""Single-frame prototypes for v41x-ex: real-meow transfer + soft-edge tail."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41h as Hh
import build_v41t_ex as T

OUT = Path('/tmp/v41x-proto')
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)
REF = Path('/workspace/ref-greenscreen/greenscreen-meow-ref2.mp4')

# Dest landmarks (stable on u-ex)
U_NOSE = np.array([718.0, 366.0], np.float32)
# upper lip stays near the philtrum; mouth centre sits under the nose
U_MOUTH_B = np.array([718.0, 418.0], np.float32)
U_MOUTH_A = np.array([718.0, 414.0], np.float32)

# Ref n250 landmarks, hand-checked on the keyed frame
# (refined after first look if needed)
REF_NOSE_250 = np.array([314.5, 140.0], np.float32)
REF_MOUTH_250 = np.array([318.0, 205.0], np.float32)


def luma(im):
    im = im.astype(np.float32)
    return 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]


def sstep(x):
    return T.sstep(x)


def load_ref_frame(i):
    cap = cv2.VideoCapture(str(REF))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    cap.release()
    assert ok, i
    return im


def key_ref(im):
    """Return rgb, fg (0-1), mic (0-1)."""
    imf = im.astype(np.float32)
    hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    green = ((H >= 35) & (H <= 85) & (S > 40) & (V > 40)) | (
        (G > R + 18) & (G > B + 15) & (G > 60)
    )
    Y = luma(imf)
    # black foam mic (specular mesh still reads mid-luma)
    mic = (Y < 95) & (~green) & (np.minimum(R, np.minimum(G, B)) < 110) & (Y < 0.72 * np.maximum(Y.mean(), 1))
    # restrict to lower half, not the mouth cavity (mouth is pinker / higher in frame)
    mic = mic & (np.arange(im.shape[0])[:, None] > 210)
    mic = mic.astype(np.uint8)
    mic = cv2.morphologyEx(mic, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mic = cv2.dilate(mic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    # keep mic in the lower-centre; don't eat the mouth cavity (also dark)
    # mouth is above the mic: suppress mic above a scan of the first large dark blob? use y gate
    # We'll refine per-frame with a flood from the bottom-most dark mass
    ys, xs = np.where(mic > 0)
    if len(ys) > 40:
        # take the component whose centroid is lowest
        n, lab, st, cent = cv2.connectedComponentsWithStats(mic, 8)
        best, by = 0, -1
        for k in range(1, n):
            cy = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] * 0.5
            area = st[k, cv2.CC_STAT_AREA]
            if area > 400 and cy > by:
                by, best = cy, k
        mic = ((lab == best) if best else mic).astype(np.uint8)
    fg = (~green).astype(np.float32)
    fg = fg * (1.0 - cv2.GaussianBlur(mic.astype(np.float32), (0, 0), 2.0))
    # erode green fringe
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fg = cv2.GaussianBlur(fg, (0, 0), 1.0)
    return imf, np.clip(fg, 0, 1), mic.astype(np.float32)


def similarity(s0, s1, d0, d1):
    """Src->dst affine. This OpenCV build's warpAffine treats M as forward
    (empirically: default flags + src_to_dst lands the point correctly)."""
    vs = s1 - s0
    vd = d1 - d0
    ls = float(np.hypot(vs[0], vs[1])) + 1e-8
    ld = float(np.hypot(vd[0], vd[1])) + 1e-8
    scale = ld / ls
    ang = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
    ca, sa = np.cos(ang), np.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], np.float32) * scale
    M = np.zeros((2, 3), np.float32)
    M[:, :2] = R
    M[:, 2] = d0 - R @ s0
    return M, scale, np.degrees(ang)


def reinhard_lab(src, dst, ms, md):
    """Colour-match src to dest on masked fur, Lab."""
    s = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.cvtColor(np.clip(dst, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    ms = (ms > 0.4)
    md = (md > 0.4)
    if ms.sum() < 30 or md.sum() < 30:
        return src
    for c in range(3):
        mu_s, sd_s = float(s[..., c][ms].mean()), float(s[..., c][ms].std()) + 1e-6
        mu_d, sd_d = float(d[..., c][md].mean()), float(d[..., c][md].std()) + 1e-6
        # slightly compress contrast toward dest, keep some source form
        out[..., c] = (s[..., c] - mu_s) * (0.55 * sd_d / sd_s + 0.45) + mu_d
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR).astype(np.float32)


def painterly(im, dest, mask):
    """Mild edge-preserving smooth + dest grain."""
    u = np.clip(im, 0, 255).astype(np.uint8)
    sm = cv2.bilateralFilter(u, 7, 28, 5)
    sm = sm.astype(np.float32)
    gdest = dest.astype(np.float32)
    grain = gdest - cv2.GaussianBlur(gdest, (0, 0), 1.15)
    out = sm + 0.70 * grain
    a = mask[..., None]
    return im * (1.0 - a) + np.clip(out, 0, 255) * a


def transfer_mouth(dst, ref_im, ref_nose, ref_mouth, dest_nose, dest_mouth, amt=1.0):
    """Warp the keyed ref muzzle onto dest and composite."""
    src, fg, mic = key_ref(ref_im)
    M, scale, ang = similarity(ref_nose, ref_mouth, dest_nose, dest_mouth)
    # source mask: muzzle + chin above the mic, not eyes
    h, w = src.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # ellipse around nose-to-mouth axis
    mid = 0.45 * ref_nose + 0.55 * ref_mouth
    # axis lengths in ref pixels
    ax, ay = 95.0, 95.0
    r2 = ((xx - mid[0]) / ax) ** 2 + ((yy - mid[1]) / ay) ** 2
    muzzle = sstep((1.0 - r2) / 0.25)
    # From the lower nose / philtrum down through the lips and visible chin,
    # stopping above the mic. Never the eyes (well above the nose).
    muzzle = muzzle * sstep((yy - (ref_nose[1] - 16.0)) / 5.0)
    muzzle = muzzle * fg

    # warp to dest
    rgb = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = cv2.warpAffine(muzzle, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(a, 0, 1)

    # fur masks for colour (exclude cavity)
    Ys = luma(src)
    cav = (Ys < 70) & (muzzle > 0.3)
    fur_s = cv2.warpAffine(((muzzle > 0.4) & (~cav) & (Ys > 80)).astype(np.float32),
                           M, (W, H), flags=cv2.INTER_LINEAR)
    Yd = luma(dst)
    # dest ginger/cream around the muzzle, not the dark slit
    fur_d = ((((Xg - dest_nose[0]) / 70.0) ** 2 + ((Yg - (dest_nose[1] + 20)) / 55.0) ** 2) < 1.0)
    fur_d = fur_d & (Yd > 70) & (Yd < 180) & (Yg > dest_nose[1] - 5) & (Yg < dest_nose[1] + 70)
    fur_d = fur_d.astype(np.float32)
    rgb = reinhard_lab(rgb, dst, fur_s, fur_d)
    rgb = painterly(rgb, dst, a)

    # extra dusk warmth: slightly lower luma toward dest mean
    if fur_s.max() > 0.3 and fur_d.max() > 0.3:
        ys = float(luma(rgb)[fur_s > 0.4].mean()) if (fur_s > 0.4).sum() > 10 else 0
        yd = float(Yd[fur_d > 0.4].mean())
        if ys > 1:
            rgb = rgb * (0.35 + 0.65 * (yd / ys))
            rgb = np.clip(rgb, 0, 255)

    # feather
    a = a * amt
    # stronger in the opening, softer on the fur fringe
    Yw = luma(rgb)
    cavity = sstep((70.0 - Yw) / 20.0) * a
    a = np.clip(0.78 * a + 0.22 * cavity, 0, 1)
    # 8-10 px fade
    binm = (a > 0.15).astype(np.uint8)
    dist_in = cv2.distanceTransform(binm, cv2.DIST_L2, 3)
    fade = sstep(dist_in / 9.0)
    a = a * fade
    # never the eyes / ears / forehead
    a = a * sstep((Yg - 330.0) / 8.0)
    a = a * sstep((dest_nose[0] + 78.0 - Xg) / 8.0)
    a = a * sstep((Xg - (dest_nose[0] - 78.0)) / 8.0)
    a = a * sstep(((dest_nose[1] + 95.0) - Yg) / 8.0)

    out = dst.astype(np.float32) * (1.0 - a[..., None]) + rgb * a[..., None]
    return np.clip(out, 0, 255), a, rgb


def eye_mask():
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (650, 272), (58, 42), -8, 0, 360, 1, -1)
    cv2.ellipse(m, (800, 278), (62, 44), 8, 0, 360, 1, -1)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 3.0), 0, 1)


def ear_mask():
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (575, 200), (70, 85), -18, 0, 360, 1, -1)
    cv2.ellipse(m, (900, 205), (68, 82), 16, 0, 360, 1, -1)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 3.0), 0, 1)


def close_old_mouth(img):
    imgf = img.astype(np.float32)
    y0, y1, x0, x1 = 378, 460, 668, 780
    sub = imgf[y0:y1, x0:x1]
    Y = luma(sub)
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    pink = (R > G + 18) & (R > B + 10) & (Y < 140) & (Y > 30)
    dark = (Y < 105) & (Y > 12) & (R + 12 > G)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ell = ((((xx - 718.0) / 38.0) ** 2) + (((yy - 422.0) / 22.0) ** 2) < 1.0)
    op = ((pink | dark | ell) & (yy > 388) & (yy < 450) & (xx > 680) & (xx < 765)).astype(np.uint8)
    op = cv2.dilate(op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if int(op.sum()) < 8:
        return img
    fill = cv2.inpaint(np.clip(sub, 0, 255).astype(np.uint8),
                       op * 255, 6, cv2.INPAINT_TELEA).astype(np.float32)
    m = cv2.GaussianBlur(op.astype(np.float32), (0, 0), 1.2)
    m = np.clip(m, 0, 1)
    out = imgf.copy()
    out[y0:y1, x0:x1] = sub * (1.0 - m[..., None]) + fill * m[..., None]
    return np.clip(out, 0, 255)


def jaw_drop(img, dy, nose, eyes, ears):
    """Lower lip + chin + upper throat move down as one piece. Tight laterally."""
    if dy < 0.3:
        return img
    imgf = img.astype(np.float32)
    cx = float(nose[0])
    # include lower lip (~408) through chin
    r2 = ((Xg - cx) / 36.0) ** 2 + ((Yg - 470.0) / 62.0) ** 2
    m = sstep((1.0 - r2) / 0.22)
    m = m * sstep((Yg - 400.0) / 8.0)
    m = m * (1.0 - sstep((Yg - 560.0) / 10.0))
    m = m * (1.0 - eyes) * (1.0 - ears)
    mapx = Xg
    mapy = (Yg - float(dy) * m).astype(np.float32)
    mapy = np.clip(mapy, 0, H - 1)
    moved = cv2.remap(imgf, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = imgf * (1.0 - m[..., None]) + moved * m[..., None]
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    # never the eyes, ears, or far cheeks
    lock = np.maximum(lock, sstep((cx - 48.0 - Xg) / 4.0))
    lock = np.maximum(lock, sstep((Xg - (cx + 48.0)) / 4.0))
    # hold the nose / upper lip (above ~398)
    lock = np.maximum(lock, 1.0 - sstep((Yg - 398.0) / 5.0))
    out = out * (1.0 - lock[..., None]) + imgf * lock[..., None]
    return np.clip(out, 0, 255)


def apply_proto_mouth(dst, ref_im, ref_nose, ref_mouth, dest_mouth, amt, dy):
    eyes, ears = eye_mask(), ear_mask()
    locked = dst.copy()
    work = close_old_mouth(dst)
    work = jaw_drop(work, dy, U_NOSE, eyes, ears)
    work, a, _ = transfer_mouth(work, ref_im, ref_nose, ref_mouth, U_NOSE, dest_mouth, amt)
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    work = work * (1.0 - lock[..., None]) + locked.astype(np.float32) * lock[..., None]
    return np.clip(work, 0, 255).astype(np.uint8), a


def crop3(im, box, sc=3):
    y0, y1, x0, x1 = box
    return cv2.resize(im[y0:y1, x0:x1], None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)


# ---------- tail proto ----------
def body_haunch_mask(mattes, tail):
    """Static body (no tail) so the haunch fur occludes the root."""
    # use a mid cycle frame whose tail is in a typical rest
    a = T.norm_alpha(mattes[0]).astype(np.float32)
    rest = tail['rest']
    # distance to rest spine
    arc, tot = Hh._arc(rest)
    s, sd, _, _ = Hh._closest(Xg, Yg, rest, arc, tot)
    w = np.interp(np.clip(s, 0, 1), np.linspace(0, 1, len(tail['widths'])), tail['widths'])
    on_tail = sstep((w + 14.0 - np.abs(sd)) / 4.0) * sstep((0.98 - s) / 0.08)
    body = np.clip(a * (1.0 - 0.96 * on_tail), 0, 1)
    # only the rear haunch / flank
    body = body * sstep((Yg - 500.0) / 14.0) * sstep((Xg - 860.0) / 14.0)
    # furry edge: slightly dilate with a little noise so the join isn't a ruler
    rng = np.random.RandomState(7)
    noise = cv2.GaussianBlur(rng.randn(H, W).astype(np.float32), (0, 0), 2.4)
    # do not add noise in the interior
    edge = cv2.GaussianBlur(body, (0, 0), 1.6)
    body = np.clip(edge + 0.08 * noise * edge * (1.0 - edge) * 4.0, 0, 1)
    return np.clip(body, 0, 1)


def fur_soft_alpha(al, s, sd, w, along):
    """Soft fur fringe instead of a hard matte."""
    rng_y = np.sin(s * 47.0) * 0.35 + np.sin(s * 113.0 + sd * 0.4) * 0.25
    fluff = 5.5 + 3.5 * rng_y
    soft = 4.5 + 2.0 * (0.5 + 0.5 * np.sin(s * 71.0))
    edge = sstep((w + fluff - np.abs(sd)) / soft)
    al = np.clip(al, 0, 1) * edge
    # rounded tip, soft
    tipz = sstep((s - 0.82) / 0.10)
    rcap = np.hypot(np.maximum(along, 0.0), sd)
    al = al * (1.0 - tipz + tipz * sstep(((w + 7.0) - rcap) / 4.0))
    # do not harden
    return np.clip(al, 0, 1)


def apply_proto_tail(img, cat_a, ph, tail, bg, occ, body, mic):
    imgf = img.astype(np.float32)
    rest = tail['rest']
    gate = T.sstep((Yg - 500.0) / 16.0) * T.sstep((Xg - 900.0) / 16.0)
    occ_s = np.clip(occ * gate, 0, 1)
    cleared = imgf * (1.0 - occ_s[..., None]) + bg * occ_s[..., None]
    # also restore plate wherever occupancy said hook but we'll only keep
    # the plate where the new tail is absent (done after drawing)

    defp = T.deform_flicks(rest, ph) if hasattr(T, 'deform_flicks') else None
    # use w-ex style deform if we inject one; for proto import from a local copy
    return cleared, defp


if __name__ == '__main__':
    print('loading frames…', flush=True)
    u = {}
    cap = cv2.VideoCapture(str(P / 'v41u-ex' / 'v41u-ex-span-890-915-lossless.mkv'))
    i = 0
    want = {0, 12, 120, 130, 138, 145, 180, 420, 425, 432, 440}
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            u[i] = im
        i += 1
    cap.release()
    print('u frames', sorted(u), flush=True)

    refs = {k: load_ref_frame(k) for k in (230, 238, 245, 250, 252, 371)}
    # save keyed debug
    for k, im in refs.items():
        src, fg, mic = key_ref(im)
        vis = src.copy()
        vis = vis * fg[..., None]
        vis[mic > 0.5] = (0, 0, 255)
        cv2.imwrite(str(OUT / f'keyed-n{k}.png'), np.clip(vis, 0, 255).astype(np.uint8))

    eyes, ears = eye_mask(), ear_mask()

    # --- mouth variants on n432 and n138 ---
    variants = []
    # A: n250 onto n432, dest mouth 418, amt 1, dy 10
    # B: slightly lower mouth 424
    # C: n250 onto n138 with less amt / smaller dest drop
    jobs = [
        ('n432-A', 432, 250, U_MOUTH_B, 1.0, 10.0),
        ('n432-B', 432, 250, np.array([718.0, 424.0], np.float32), 1.0, 12.0),
        ('n432-C', 432, 250, np.array([718.0, 414.0], np.float32), 0.95, 8.0),
        ('n138-A', 138, 238, U_MOUTH_A, 0.85, 6.5),
        ('n138-B', 138, 245, U_MOUTH_A, 0.90, 7.0),
        ('n138-C', 138, 250, U_MOUTH_A, 0.70, 6.0),
    ]
    # ref landmarks: start with n250, assume similar for nearby (head is stable in 10s peak)
    ref_lm = {
        250: (np.array([314.5, 140.0], np.float32), np.array([318.0, 205.0], np.float32)),
        252: (np.array([314.5, 140.0], np.float32), np.array([318.0, 206.0], np.float32)),
        245: (np.array([314.5, 142.0], np.float32), np.array([318.0, 200.0], np.float32)),
        238: (np.array([314.5, 144.0], np.float32), np.array([318.0, 195.0], np.float32)),
        230: (np.array([314.5, 146.0], np.float32), np.array([318.0, 188.0], np.float32)),
        371: (None, None),  # fill after
    }
    # quick auto mouth/nose on 371
    im = refs[371]
    src, fg, mic = key_ref(im)
    # reuse earlier detector loosely — mark and we'll see
    ref_lm[371] = (np.array([400.0, 160.0], np.float32), np.array([410.0, 230.0], np.float32))

    for name, ui, ri, dm, amt, dy in jobs:
        out, a = apply_proto_mouth(u[ui], refs[ri], ref_lm[ri][0], ref_lm[ri][1], dm, amt, dy)
        box = (330, 520, 610, 870)
        before = crop3(u[ui], box)
        after = crop3(out, box)
        # alpha vis
        av = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        av = cv2.cvtColor(av, cv2.COLOR_GRAY2BGR)
        av = crop3(av, box)
        pair = np.hstack([before, after, av])
        cv2.putText(pair, f'u {name}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.putText(pair, f'x {name}', (before.shape[1] + 8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        path = OUT / f'mouth-{name}.png'
        cv2.imwrite(str(path), pair)
        print('wrote', path, flush=True)
        cv2.imwrite(str(OUT / f'mouth-{name}-full.png'), out)

    print('mouth proto done', flush=True)
