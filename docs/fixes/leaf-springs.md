# Plan: leaf springs

Issue #4. This commit is the plan only. The spring code is being developed elsewhere and will be pushed to `fix/leaf-springs` later.

## What is going wrong

The leaves are a single static image. They do not bend when the tail brushes them, and they do not sway on their own. Leaf-colored pixels in cycle 0 move about 0.0003 px per frame, which is effectively still. There is no joint, no pivot, and no response to the tail.

## What to change

Treat each front leaf as a semi-rigid card that rotates about the stem where it meets the pot. Use the same cards and pivots as the tail-layer plan (issue #2). The leaf has a spine, so it rocks about that pivot. It does not ripple like cloth.

- Leaf 7, the large mid leaf, pivots near (1184, 654) and bends at most 12°.
- Leaf 3, the upper leaf, pivots near (1171, 577) and bends at most 8°.
- Leaf 10 stays behind the tail and can stay still, or take a tiny ambient sway of about 3°.
- The right-hand foliage (components 2, 4, 5, 9) stays still or shares a very small sway, up to about 4°.

Spring settings, per frame, for a gentle lo-fi sway. Period about 14 frames (about 0.58 s at 24 fps) and damping ratio 0.60, so the leaf returns with one or two soft wobbles and then settles. That pair is stiffness `k` about 0.20 and damping `c` about 0.54.

Each frame, with `dt = 1`:

- `omega += (-k * theta - c * omega + contact_torque) * dt`
- `theta += omega`
- clamp `theta` to the max for that leaf

Drive `contact_torque` from the generated tail, using the cream tail matte of the current frame before the static plant is overlaid. A signed distance from the leaf blade to the tail, plus the tail's velocity along the leaf normal, is enough. When the tail is inside about 28 px, push the leaf away from the tail's path. When the tail leaves, the spring brings the leaf back with a small overshoot and settles in about 16–20 frames.

Add a very small idle sway so the plant never looks frozen, underneath the contact response.

Keep the motion 241-periodic: run two cycles of warmup, then keep one cycle, so the loop join stays put.

## How to check

During a tail flick, leaf 7's flow is no longer zero: the leaf yields and then settles. Leaf 3 follows more quietly. The tail still reads as a solid limb between the layers from the tail-leaf plan, with the soft contact shadow on the front leaves.
