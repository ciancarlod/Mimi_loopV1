# Mimi loop

Lo-fi looping video of a cat named Mimi. The picture is a 1280×720, 24 fps composite: a painted room plate, generated cat clips, and a 241-frame cycle joined into a 25-second test. The latest builder is `scripts/build_v41ah_ex.py` (`v41ah-ex`).

## Assets are not in this repo

The source plates, generated clips, mattes, and renders are about **42 GB**. They stay off-repo, under `/workspace/lofi-v41-meow` on the machine that builds the loop (older rounds also read `/workspace/lofi-v30-final`, `/workspace/lofi-v30`, `/workspace/lofi-v28`, `/workspace/lofi-v24`, and `/workspace/ref-greenscreen`). Nothing in that tree is committed. `.gitignore` blocks video, image, audio, and numpy-array outputs, plus `qc/` and `genvid/`.

To render in CI later, put the plates and clips in an `assets/` folder in the repo, or attach them to a GitHub release as `mimi-assets.tar.gz`. The current scripts will still not read that folder until their hard-coded `/workspace` paths take an asset root. See `.github/workflows/render-test-frames.yml`.

## Layout

- `scripts/` — the 43 Python builders and helpers, original file names.
- `docs/FINDINGS.md` — physics notes for the head smear, the tail-through-leaf cut, the mouth close, and leaf springs.
- `requirements.txt` — `numpy` and `opencv-python-headless`. `ffmpeg` is also required.

`scripts/precompute_v41f_mattes.py` additionally imports Pillow and `rembg`, which are not in `requirements.txt`. It reads a lossless span and is not part of the import check.

## How to run

Install dependencies, and install ffmpeg from your system packages.

```bash
python -m pip install -r requirements.txt
```

The builders set `P = Path('/workspace/lofi-v41-meow')` and put that directory first on `sys.path`. Data has to be there. Launching a script as a file still finds the other builders in `scripts/` when they are not copied into the asset tree.

Latest round, preview QC stills (a large frame set, not the full encode):

```bash
python scripts/build_v41ah_ex.py --preview
```

Full 25-second encode. This refuses to overwrite an existing `v41ah-ex-test-25s.mp4`:

```bash
python scripts/build_v41ah_ex.py
```

Optional mouth reference:

```bash
python scripts/build_v41ah_ex.py --mouth-ref /path/to/ref.mp4
```

`scripts/build_v41ah_ex.py --preview` still expects the frozen u-ex plates, the prior encodes `v41u-ex` through `v41af-ex`, the PCM stem, and the `v41ab-ex` mattes (`g1v3` / `g2v3`) under `/workspace/lofi-v41-meow`. It writes PNGs to `/workspace/lofi-v41-meow/qc/v41ah-ex/preview`.

Older full-span renderer. It has no arguments. Running it imports `compositor` from `/workspace/lofi-v30-final`, loads the v41 cycle caches, and encodes frames `890*24` through `915*24`:

```bash
python scripts/render_v41_span.py
```

QC crops from an encode that already exists in the asset tree (`verify_round.py` reads the mp4 as soon as it loads):

```bash
python scripts/verify_round.py v41ah-ex
```

`scripts/mix_v41_audio.py` also runs on import and rewrites the audio stem under the asset tree. Do not import it unless those stems are in place.
