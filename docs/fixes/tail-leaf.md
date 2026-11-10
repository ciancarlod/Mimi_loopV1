# Plan: tail through the front leaves

Issue #2. This commit is the plan only. The layering changes are being developed elsewhere and will be pushed to `fix/tail-leaf` later.

## What is going wrong

The front leaves are one static card. `u_plant_matte()` builds a single filled silhouette from the u-ex leaf shapes, and `compose_layered` pastes that card over the whole cat and tail. The tail is cut out wherever the card says "leaf", and the leaf does not rotate or give way. Gaps left in the filled matte show cream tail in the middle of the foliage, which reads as the tail punching a hole through a leaf.

Leaf-colored pixels in the current loop barely move (median flow about 0.0003 px per frame). The plant in the composite matches the static plate.

An earlier speckle bug, where a per-frame green gate treated dark tail fur as leaf, is already fixed by dropping that gate. This plan leaves that gate out. The remaining problem is the solid static card and the missing layer order.

## What to change

Split the plant from the labeled u-ex leaf components and stack the layers so the tail slides between them. Keep the current solid-tail inpaint.

| Component | Role | Stem pivot (pot side) | Max rotation |
|---|---|---|---|
| 7 | Front, the large mid leaf the tail crosses | about (1184, 654), not the free tip at (1041, 695) | 12° |
| 3 | Front, upper leaf near the tail tip | about (1171, 577) | 8° |
| 10 | Behind the tail | about (1218, 693) | still, or a tiny ambient sway |
| 2, 4, 5, 9 | Right-hand pot foliage, behind or off the tail path | pot-ward pivots | still, or a very small shared sway |

Paint order, per pixel:

1. Back leaves (2, 4, 5, 9, 10).
2. Cat and tail.
3. Front leaves 7 and 3, each an RGBA card cropped from a quiet u-ex frame, feathered 1–2 px, rotated about its stem with `cv2.warpAffine`.
4. A soft contact shadow of the tail on leaves 7 and 3: dilate the overlap about 8–12 px, blur it, and darken the front leaf slightly at the contact.

Drive the rotation with the damped spring in the leaf-springs plan, using the generated tail contour from the current frame before the plant is overlaid. This branch's job is the segmentation, the layer order, and the contact shadow. The spring itself is issue #4.

## How to check

At 3×, tail strips around frames 60–92 and 170–210: leaf 7 yields and recovers, and the tail stays a solid limb that disappears between leaves rather than through a hole in one card.
