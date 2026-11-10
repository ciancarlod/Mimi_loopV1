import cv2, numpy as np, json, subprocess as sp
from pathlib import Path
P = Path('/workspace/lofi-v41-meow')
prm = json.load(open(P / 'cycle' / 'v41-medium-params.json'))
eboxes = {r['frame']: r['expanded_box'] for r in prm['records']}
# growl params may have different eboxes? same _ebox function, same boxes
v40 = cv2.VideoCapture('/workspace/lofi-v40-meow/v40-test-25s.mp4')
v41 = cv2.VideoCapture(str(P / 'v41-test-25s.mp4'))
NFR = 600

def style_label(i, t, ver):
    if 123 <= i <= 151: tag = 'OPEN mid'
    elif 179 <= i <= 207: tag = 'MRRRP'
    elif 364 <= i <= 392: tag = 'GROWL'
    elif 420 <= i <= 448: tag = 'OPEN big'
    else: tag = ''
    return f'{ver} {tag} {t:.2f}s' if tag else f'{ver} {t:.2f}s'

def face_panel(frame, ebox, label, w=640, h=720):
    xa, ya, ew, eh = ebox
    # whole face: most of the ebox, slightly cropped, 3x
    cw, ch = w // 3, h // 3
    cx = xa + ew // 2
    cy = ya + int(eh * 0.42)
    xa2 = int(np.clip(cx - cw / 2, 0, frame.shape[1] - cw))
    ya2 = int(np.clip(cy - ch / 2, 0, frame.shape[0] - ch))
    crop = frame[ya2:ya2 + ch, xa2:xa2 + cw]
    if crop.shape[0] != ch or crop.shape[1] != cw:
        canvas = np.zeros((ch, cw, 3), np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    z = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.72, 1
    (tw, th), bas = cv2.getTextSize(label, font, scale, thick)
    pad_x, pad_y = 8, 6
    x1, y1 = 10, 10
    x2, y2 = x1 + tw + 2 * pad_x, y1 + th + bas + 2 * pad_y
    cv2.rectangle(z, (x1, y1), (x2, y2), (16, 14, 12), -1)
    cv2.putText(z, label, (x1 + pad_x, y1 + pad_y + th),
                font, scale, (240, 240, 240), thick, cv2.LINE_AA)
    return z

raw = P / 'work' / 'v40-vs-v41-whole-face-ba.raw'
# stream via ffmpeg from stdin video + v41 pcm
enc = sp.Popen([
    'ffmpeg', '-v', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '24', '-i', '-',
    '-f', 'f32le', '-ar', '48000', '-ac', '2', '-i', str(P / 'work' / 'test-audio-890-915-v41.f32le'),
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
    '-shortest', '-movflags', '+faststart',
    str(P / 'v40-vs-v41-whole-face-before-after.mp4')
], stdin=sp.PIPE)
nfr = 0
for i in range(NFR):
    ok40, f40 = v40.read(); ok41, f41 = v41.read()
    assert ok40 and ok41, i
    ph = (890 * 24 + i) % 241
    ebox = eboxes.get(ph, [540, 266, 250, 272])
    t = i / 24
    pair = np.concatenate([
        face_panel(f40, ebox, style_label(i, t, 'v40')),
        face_panel(f41, ebox, style_label(i, t, 'v41'))
    ], 1)
    enc.stdin.write(pair.tobytes())
    nfr += 1
enc.stdin.close(); assert enc.wait() == 0
print('wrote whole-face before-after', nfr, 'frames')
