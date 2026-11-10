# Plan: head smear

Issue #1. This commit is the plan only. The compositing changes are being developed elsewhere and will be pushed to `fix/head-smear` later.

## What is going wrong

Two problems show up together when Mimi's head moves.

The background plate still contains the painted rest-pose cat. `compose_warped` (in the ae-ex builder, and still the same idea in later rounds) composites the generated cat at the rest pose, then mesh-warps that plate. The warp is gated by the unwarped cat silhouette and is filled, behind the moved head, with the raw plate. Where the warped alpha does not cover the old position, a 1–3 px fringe of the old outline stays behind the ear and the eye highlight. At a close crop that fringe reads as a trailing double edge. It is leftover paint from the plate.

On top of that, the head path is a series of long holds. Of the 241 frames in a cycle, about 167 are authored holds and about 66 are eases. The head eases a pixel or two, sits still for a second or more, then eases again. Forehead motion during the longest holds is a few hundredths of a pixel per frame, so the turn feels like a slideshow.

Interpolating extra frames will not remove the fringe. The ghost is already in the plate, at 24 fps.

## What to change

Work in `compose_warped` and the head path, on a new version name so ae-ex, af-ex, and ag-ex stay frozen. Keep audio, the scrolling title, snow, and steam as they are.

1. Build one cat-free underlay. Heal the painted cat out of the u-ex plate in the head, chest, and ears, and leave the window, wall, and plants. Fill behind a moving head from that underlay.
2. Warp the composited cat and its alpha together. Set the warp gate to the union of the unwarped silhouette and the predicted moved silhouette, dilated a little past the largest displacement. Keep the window pane locked only on pixels that are not cat at either end of the warp.
3. Replace the dead holds with one 241-frame head path that starts and ends at rest, moves on an arc, and slows down at changes of direction. A hold may only be a moving hold: a little residual drift, the existing lagged breath (`HEAD_LAG` of 8 frames can stay), and ears and whiskers that settle over about 6–10 frames after the skull. Shoulders lead and the head follows. Keep the motion gentle. Avoid a freeze longer than about 12 frames, except the last 8 seam frames.
4. After the silhouette is a single clean edge, a short directional blur can ride the fast moves (shutter about a third to a half of a frame, only where the warp moves more than about 0.4 px per frame). Take it from the warp field.

## How to check

At 6×, the left ear and the left eye highlight over the ease around frames 54–70 should be one sharp silhouette, with no muddy fringe at the rest pose. During the old hold stretches the forehead should still drift a little, rather than sitting frozen for a full second.
