#!/usr/bin/env python3
"""v41 audio: keep v41-eyesonly (v40) PCM except the 15.4s window.
Remove the CC0 purr and insert one closed-mouth meow-growl from
/workspace/ref-greenscreen/ref4.mp4, level-matched to the -3 dB meow stem.
"""
import hashlib, json, subprocess as sp
from pathlib import Path
import numpy as np

P = Path('/workspace/lofi-v41-meow')
FS = 48000
EYES = P / 'work' / 'test-audio-890-915-v41-eyesonly.f32le'
ORIG = Path('/workspace/lofi-v40-meow/work/test-audio-890-915-meow-3dB.f32le')
OUT = P / 'work' / 'test-audio-890-915-v41.f32le'
QWAV = Path('/workspace/lofi-v25/quiet-user-meows-only.wav')
GAIN = 10 ** (-3 / 20)
PURR = Path('/workspace/lofi-v40-meow/qc/audio/purr-loop-hq.f32le')

eyes = np.fromfile(EYES, np.float32).reshape(-1, 2)
orig = np.fromfile(ORIG, np.float32).reshape(-1, 2)
assert eyes.shape == orig.shape == (25 * FS, 2)

q = np.frombuffer(sp.check_output([
    'ffmpeg', '-v', 'error', '-i', str(QWAV),
    '-ar', '48000', '-ac', '2', '-f', 'f32le', '-'
]), np.float32).reshape(-1, 2)
off = (890 * FS) % len(q)
idx = (off + np.arange(len(eyes))) % len(q)
stem = q[idx] * np.float32(GAIN)

T_LOOP0 = 89 / 24.0
CYCLE = 241 / 24.0

def span_window(loop_t0, loop_t1):
    return T_LOOP0 + loop_t0, T_LOOP0 + loop_t1

# Old purr window (v40 / v41-eyes): second play of meow A
w_purr = span_window(1.680 + CYCLE, 2.330 + CYCLE)
# New growl window: slightly longer to fit the ref4 slice, contains the old window
w_growl = (15.400, 16.250)
print('w_purr', w_purr, 'w_growl', w_growl)

def fade_mask(n, t0, t1, fade_s=0.016):
    fade = int(round(fade_s * FS))
    w = np.zeros(n, np.float32)
    a, b = int(round(t0 * FS)), int(round(t1 * FS))
    a = max(0, a); b = min(n, b)
    w[a:b] = 1.0
    if fade > 0 and b - a > 2 * fade:
        w[a:a+fade] *= np.linspace(0, 1, fade, endpoint=False, dtype=np.float32)
        w[b-fade:b] *= np.linspace(1, 0, fade, endpoint=False, dtype=np.float32)
    return w, a, b

n = len(eyes)
m_purr, ap, bp = fade_mask(n, *w_purr, fade_s=0.012)  # exact v40 fade
m_gr, ag, bg = fade_mask(n, *w_growl, fade_s=0.016)

# Reconstruct the CC0 purr that was added in v40 (same recipe)
purr_all = np.fromfile(PURR, np.float32).reshape(-1, 2)
len3 = bp - ap
purr_start = int(2.0 * FS)
pu = np.zeros((len3, 2), np.float32)
sl = purr_all[purr_start:purr_start + len3]
pu[:len(sl)] = sl

def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))

ref3 = stem[ap:bp]
scale3 = rms(ref3) / max(rms(pu), 1e-8)
peak_ref = float(np.abs(ref3).max())
peak_new = float(np.abs(pu).max())
if peak_new * scale3 > peak_ref * 1.02:
    scale3 = peak_ref * 1.02 / max(peak_new, 1e-8)
pu *= np.float32(scale3)
print('reconstructed purr scale', scale3)

# Load ref4, slice 0.95-1.80s (closed-mouth meow-growl, mouth sealed)
ref4 = np.frombuffer(sp.check_output([
    'ffmpeg', '-v', 'error', '-i', '/workspace/ref-greenscreen/ref4.mp4',
    '-ar', '48000', '-ac', '2', '-f', 'f32le', '-'
]), np.float32).reshape(-1, 2)
src0, src1 = 0.95, 1.80
s0, s1 = int(round(src0 * FS)), int(round(src1 * FS))
clip = ref4[s0:s1].copy()
# tiny 4ms edge soften on the source itself to avoid clicks before the span fade
edge = int(0.004 * FS)
if edge > 0 and len(clip) > 2 * edge:
    clip[:edge] *= np.linspace(0, 1, edge, endpoint=False, dtype=np.float32)[:, None]
    clip[-edge:] *= np.linspace(1, 0, edge, endpoint=False, dtype=np.float32)[:, None]

len_g = bg - ag
placed = np.zeros((len_g, 2), np.float32)
# center the 0.85s clip in the 0.85s window (15.40-16.25 is 0.85s — exact fit)
if len(clip) >= len_g:
    # trim equally
    extra = len(clip) - len_g
    placed[:] = clip[extra//2: extra//2 + len_g]
    used = (src0 + (extra//2)/FS, src0 + (extra//2 + len_g)/FS)
else:
    off = (len_g - len(clip)) // 2
    placed[off:off+len(clip)] = clip
    used = (src0, src1)
print('ref4 used', used, 'placed samples', len_g, 'clip', len(clip))

# Level-match to the -3 dB meow stem in the old purr/meow-A window (not louder)
ref_g = stem[ap:bp]  # original meow level reference
scale_g = rms(ref_g) / max(rms(placed), 1e-8)
peak_ref = float(np.abs(ref_g).max())
peak_new = float(np.abs(placed).max())
if peak_new * scale_g > peak_ref * 1.02:
    scale_g = peak_ref * 1.02 / max(peak_new, 1e-8)
placed *= np.float32(scale_g)
print('growl scale', scale_g,
      'ref_rms_db', 20*np.log10(rms(ref_g)+1e-12),
      'new_rms_db', 20*np.log10(rms(placed)+1e-12),
      'ref_peak_db', 20*np.log10(peak_ref+1e-12),
      'new_peak_db', 20*np.log10(float(np.abs(placed).max())+1e-12))

# Build output from eyesonly:
# 1) subtract the reconstructed CC0 purr in the old window
# 2) in any *new* samples outside the old window, also remove the meow stem
# 3) add the ref4 growl with the new fade mask
new = eyes.copy()
new[ap:bp] -= pu * m_purr[ap:bp, None]

# extra region: in m_gr but not in the original purr replacement.
# Those samples still have music+meow_stem (eyes == orig there).
extra = (m_gr > 0) & (m_purr == 0)
new -= stem * extra[:, None]

growl_track = np.zeros_like(new)
growl_track[ag:bg] = placed
new += growl_track * m_gr[:, None]

# Proof vs eyesonly: only the union of the two 15.4s windows may change
changed = np.any(new != eyes, axis=1)
union = (m_purr > 0) | (m_gr > 0)
ident_out = bool(np.array_equal(new[~union], eyes[~union]))
max_out = float(np.abs(new[~union] - eyes[~union]).max()) if not ident_out else 0.0
# also confirm no change in the mrrrp window
m_mrrrp, a2, b2 = fade_mask(n, *span_window(4.000, 4.690), fade_s=0.012)
ident_mrrrp = bool(np.array_equal(new[a2:b2], eyes[a2:b2]))
print('outside 15.4s union identical to eyesonly', ident_out, 'max_out', max_out)
print('mrrrp window identical', ident_mrrrp)
print('changed samples', int(changed.sum()), 'union samples', int(union.sum()))
print('changed t range', changed.argmax()/FS if changed.any() else None,
      (len(changed) - 1 - changed[::-1].argmax())/FS if changed.any() else None)

# clicks: look at first difference of peak in fade regions
d = np.diff(new, axis=0)
print('max |delta| overall', float(np.abs(d).max()),
      'in fade-in', float(np.abs(d[ag:ag+int(0.02*FS)]).max()) if bg>ag else None,
      'in fade-out', float(np.abs(d[bg-int(0.02*FS):bg]).max()) if bg>ag else None)

new.astype(np.float32).tofile(OUT)
h = hashlib.sha256(new.astype(np.float32).tobytes()).hexdigest()
eyes_h = hashlib.sha256(eyes.astype(np.float32).tobytes()).hexdigest()
meta = {
    'eyesonly_pcm_sha256': eyes_h,
    'new_pcm_sha256': h,
    'samples': int(n),
    'windows_s': {'old_purr_removed': list(w_purr), 'growl_added': list(w_growl)},
    'fade_s': 0.016,
    'outside_15.4s_union_byte_identical_to_v41_eyesonly': ident_out,
    'max_abs_diff_outside': max_out,
    'mrrrp_window_byte_identical_to_v41_eyesonly': ident_mrrrp,
    'changed_samples': int(changed.sum()),
    'source': {
        'file': '/workspace/ref-greenscreen/ref4.mp4',
        'description': 'Mimi closed-mouth meow-growl, lips sealed, no mic over the mouth',
        'source_timestamps_s': [src0, src1],
        'source_timestamps_used_s': list(used),
        'placed_at_clip_s': list(w_growl),
        'level_match': 'scaled to the -3 dB original meow stem, peak-capped at 1.02x (not louder)',
        'scale': scale_g,
    },
    'open_meow_stems_and_mrrrp_kept': True,
}
json.dump(meta, open(P / 'work' / 'audio-v41.json', 'w'), indent=1)
print('wrote', OUT, 'sha', h)
assert ident_out and ident_mrrrp and max_out == 0.0
