import sys, json, numpy as np, cv2, subprocess as sp, hashlib
sys.path.insert(0, '/workspace/lofi-v30-final')
import compositor as c
from pathlib import Path
P = Path('/workspace/lofi-v41-meow')
T0, T1 = 890, 915
a, b = T0 * 24, T1 * 24
v30frames = list(c.frames)
styles = {
    'medium': np.load(P / 'cycle' / 'v41-medium-cycle.npy'),
    'mrrrp':  np.load(P / 'cycle' / 'v41-mrrrp-cycle.npy'),
    'growl':  np.load(P / 'cycle' / 'v41-growl-cycle.npy'),
    'big':    np.load(P / 'cycle' / 'v41-big-cycle.npy'),
}
# Use medium params for eboxes (same _ebox)
prm = json.load(open(P / 'cycle' / 'v41-medium-params.json'))
active = set(prm['active_frames'])
eboxes = {r['frame']: r['expanded_box'] for r in prm['records']}
for _n in range(119, 124):
    if _n not in eboxes and 118 in eboxes:
        eboxes[_n] = eboxes[118]

# mrrrp ease-out continues a few frames past the v31 active window
# (ph 119-123 -> clip i 208-212, ~8.67-8.83s) so the close cannot snap at 8.66s.
MRRRP_I = set(range(179, 213))

def style_for(global_n):
    i = global_n - a
    ph = global_n % 241
    if ph not in active and i not in MRRRP_I:
        return None, ph
    # Two full cycles in the 25s span after i=89 and i=330
    # First meow-1 window: i 123-151  -> medium
    # First meow-2 window: i 179-212  -> mrrrp (includes 4+ frame ease-out)
    # Second meow-1: i 364-392        -> closed-mouth growl (ref4)
    # Second meow-2: i 420-448        -> big
    if 123 <= i <= 151:
        return 'medium', ph
    if i in MRRRP_I:
        return 'mrrrp', ph
    if 364 <= i <= 392:
        return 'growl', ph
    if 420 <= i <= 448:
        return 'big', ph
    return None, ph

# identity of caches vs v30 outside ebox
for st, arr in styles.items():
    for n in range(241):
        extra = (st == 'mrrrp' and n in range(119, 124))
        if n not in active and not extra:
            assert np.array_equal(arr[n], v30frames[n]), (st, n)
        elif n in eboxes or extra:
            x0, y0, w, h = eboxes[n]
            m = np.ones((720, 1280), bool)
            m[y0:y0 + h, x0:x0 + w] = False
            assert np.array_equal(arr[n][m], v30frames[n][m]), (st, n)

lossless = sp.Popen([
    'ffmpeg', '-v', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '24', '-i', '-',
    '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0',
    str(P / 'work' / 'v41-span-890-915-lossless.mkv')
], stdin=sp.PIPE)
master = sp.Popen([
    'ffmpeg', '-v', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '24', '-i', '-',
    '-f', 'f32le', '-ar', '48000', '-ac', '2', '-i', str(P / 'work' / 'test-audio-890-915-v41.f32le'),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-movflags', '+faststart', '-shortest',
    str(P / 'v41-test-25s.mp4')
], stdin=sp.PIPE)
stats = []
h30 = hashlib.sha256(); h40 = hashlib.sha256()
for n in range(a, b):
    c.frames = v30frames; f30 = c.render(n)
    st, ph = style_for(n)
    if st is None:
        f40 = f30
    else:
        c.frames = list(styles[st]); f40 = c.render(n)
    d = np.any(f30 != f40, axis=2)
    cnt = int(d.sum())
    inbox = True
    box_used = None
    if cnt:
        ys, xs = np.nonzero(d)
        x0, y0, w, h = eboxes[ph]
        box_used = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
        inbox = bool(xs.min() >= x0 and xs.max() < x0 + w and ys.min() >= y0 and ys.max() < y0 + h)
    stats.append({
        'global_frame': n, 't_s': n / 24, 'phase': ph, 'style': st,
        'meow_active_phase': (ph in active) or (st == 'mrrrp'), 'changed_pixels': cnt,
        'all_changes_inside_expanded_box': inbox,
        'changed_bbox': box_used
    })
    assert (ph in active) or (st == 'mrrrp' and 208 <= (n - a) <= 212) or cnt == 0
    assert inbox
    lossless.stdin.write(f40.tobytes())
    master.stdin.write(f40.tobytes())
    h30.update(f30.tobytes()); h40.update(f40.tobytes())
    if (n - a) % 100 == 0:
        print('rendered', n, st, flush=True)
for e in (lossless, master):
    e.stdin.close(); assert e.wait() == 0
json.dump({
    'span': [T0, T1], 'frames': b - a,
    'nonmeow_frames_identical': sum(1 for s in stats if not s['meow_active_phase'] and s['changed_pixels'] == 0),
    'nonmeow_frames': sum(1 for s in stats if not s['meow_active_phase']),
    'meow_frames': sum(1 for s in stats if s['meow_active_phase']),
    'all_changes_inside_expanded_boxes': all(s['all_changes_inside_expanded_box'] for s in stats),
    'v30_render_sha256': h30.hexdigest(),
    'v41_render_sha256': h40.hexdigest(),
    'assignment': {'5.4s': 'medium', '7.7s': 'mrrrp', '15.4s': 'growl', '17.75s': 'big'},
    'per_frame': stats
}, open(P / 'work' / 'span-890-915-identity.json', 'w'), indent=0)
print('done', h30.hexdigest(), h40.hexdigest())
