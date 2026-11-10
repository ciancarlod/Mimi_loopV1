# Mimi lo-fi loop — physics diagnosis (v41ae-ex / v41af-ex)

Read-only study. No new render, and no existing project files were modified. Evidence and plots live under `/workspace/lofi-v41-meow/physics-dive/`.

**Subjects.** Delivered loop: `v41ae-ex-test-25s.mp4` (600 frames, 1280×720, 24 fps, 241-frame cycle). Latest sibling encode: `v41af-ex-test-25s.mp4` (blinks added; same mouth / tail / head-warp pipeline). `v41ag-ex` was being built by another worker and was not measured. Builders consulted (not edited): `build_v41ae_ex.py`, `build_v41af_ex.py`, plus `build_v41z_ex.py` / `build_v41y_ex.py` for retiming, seam morph, and `src_time`.

**Method.** Encoded frames were extracted with OpenCV and stepped at 3× (and 6× for the left ear and muzzle). Head motion was measured with phase correlation and DIS flow in a tight face crop, then re-measured on a forehead crop that excludes the muzzle (the first pass was contaminated by the meow). Mouth opening was measured as the longest dark/pink run in a center band (`x=700–750`, `y=360–450`), because a global luma threshold on this dim plate classifies cream fur as cavity. Plant motion was measured as optical flow on leaf-colored pixels and by comparing the composite to `genvid/u-n0.png` and to `genvid/gen2.mp4`.

---

## 1. Head smear / leftover-frame / slideshow lag

### What the encoded frames actually show

Two separate problems are stacked. They are easy to conflate because both appear when the head “moves.”

**A. Slideshow timing (the laggy, not-smooth feel).**  
Head translation and tilt are not a continuous per-frame performance. `build_v41ae_ex.py` uses settle-beat knots — ease, *hold*, ease — with smootherstep only on the moves:

| Local frames | Kind | Pose (dx, dy, tilt°) | Duration |
|---|---|---|---|
| 0–14 | hold | (0, 0, 0) | 14 f (0.58 s) |
| 14–30 | ease | → A (1.10, −0.80, 0.40) | 16 f |
| 30–54 | hold A | | 24 f (1.00 s) |
| 54–70 | ease | → B (−1.25, −1.35, −0.35) | 16 f |
| 70–98 | hold B | | 28 f (1.17 s) |
| 98–114 | ease | → C (0.55, −0.45, 0.25) | 16 f |
| 114–150 | hold C | | 36 f (1.50 s) |
| 150–168 | ease home | → (0, 0, 0) | 18 f |
| 168–241 | hold 0 (incl. 8-frame seam at the end) | | 73 f (3.04 s) |

Of 241 cycle frames, **167 are authored holds and only 66 are eases** (plus a little breath, ear twitch, and meow anticipation on top). Knot amplitudes are only about 1–2 px of translation and <0.5° of tilt, plus 1.2 px of lagged breath and a 2.4 px pre-meow lift.

Forehead-only per-frame phase (mouth excluded), cycle 0, median |step|:

- ease to A / to B: **0.23 px/frame**
- hold B (70–98): **0.038 px/frame** — effectively frozen
- hold 0 before the seam (168–233): **0.062 px/frame**
- last-8-frame seam morph (233–240): **0.041 px/frame** (smooth, small)
- hold A (30–54): 0.37 px/frame — *not* frozen; breath and/or gen-clip residual still move the forehead here
- hold C (114–150): large steps (median 1.0, max 3.5) are the meow / anticipation, not a head teleport

Plot: `plots/head_displacement_cycle0.png`, `plots/head_forehead_ear_cycle0.png`. Table: `head_displacement.csv`, `head_forehead_ear.csv`.

So the “slideshow shifting” read is real and authored: the head eases to a pose, sits there for a second or more, then eases again. That is a moving-hold used as a *dead* hold, not a moving hold with follow-through.

**B. Trailing smear / leftover rest-pose silhouette.**  
This is a compositing bug, not motion blur and not (primarily) a ghost blend of two time instants.

Pipeline, from `compose_layered` then `compose_warped` in `build_v41ae_ex.py`:

1. The gen cat is composited onto the u-ex plate at the *unwarped* rest pose, producing `plate`.
2. A mesh warp `(dx, dy)` is gated by a **dilated unwarped cat matte** (`dilate` 9 px + Gaussian 2.0) and is zeroed on the window pane.
3. `compose_warped` remaps `plate` and the cat alpha with that gated field. Behind the *unwarped* silhouette it fills with `u` — the raw u-ex plate, **which still contains the painted rest-pose cat**. Where the warped alpha `a_w` does not fully cover the old position, that rest-pose head remains in the output.

Because the authored displacement is only 1–3 px and the gate is ~9 px, the moved head never leaves the unwarped silhouette, so you do not get a clean 10 px double. You get a **1–3 px leftover fringe**: a muddy second contour sitting on the previous position. At 6× that reads as a trailing smear, exactly as described (“leftover frame, like a slideshow shifting”).

Evidence:

- `head/easeB-n070-Lear6x.png` — single frame at the end of the largest ease. The left-ear silhouette against the window has a soft, muddy fringe, not a hard painterly edge.
- `head/easeB-n070-Lear6x-overlay.png` and `head/easeB-n070-Leye6x-overlay.png` — n14 (rest) in red vs n70 (moved) in cyan. The ear and the eye highlight have clearly shifted; a leftover at the red position in the *same* n70 frame would sit right under that fringe. The same pattern is in v41af-ex (`head/af-easeB-n070-Lear6x-overlay.png`).
- `head/move-ease_to_B-n54-70-face3x-strip.png` and `head/move-ease_to_B-consec-diff8x-strip.png` — the move is a short ease between two holds.
- `head/move-ease_to_B-n54-70-overlay-rest-red-moved-cyan-strip.png` — rest vs moved dipole during the ease.

The warp gate cannot be the *only* limiter of motion (the 9 px dilate is larger than the 2 px move), but it *is* why a leftover at the old position is possible: `dx, dy` are zero outside the unwarped cat, so the destination pixels of a larger move would not sample the cat at all. Any future, bigger 3D-looking head turn will break unless the gate is warped with the head or is the union of old and new silhouettes.

### Checked and ruled out as the main smear

- **Exact duplicated encoded frames.** Consecutive tight-face MAD never hits zero (snow, steam, grain, lamp). “Near-duplicate” flags (MAD < 0.35) cluster on *authored holds* (16 of them in hold B), which is the hold itself, not a codec drop.
- **Cycle-end DIS flow morph (last 8 frames of each cycle).** Seam step medians are 0.04–0.07 px. No special ghost appears in `head/seam-n228-245-face3x-strip.png`. This is not what happens during the mid-cycle head eases.
- **Midpoint-blend / dissolve ghosts as a general mechanism.** A naive “closer to 0.5×(prev+next) than to either neighbor” flag fired on hundreds of frames, including holds, and is not diagnostic of a dissolve. The seam path uses an occlusion-aware flow morph (“No dissolve ghost” in `build_v41z_ex.py`), and mid-cycle head motion is a remap of a single plate, not a 50/50 mix of two times.
- **Gen-clip frame duplication as the smear.** `gen1.mp4` / `gen2.mp4` do have long near-still stretches (gen1: 43% of steps < 0.05 px, 98 near-dups; gen2: 25% / 56). That can add to the dead-hold feeling, but the leftover fringe is produced *after* the gen cat is already on the plate, by `compose_warped`.
- **The 30 px “head jump” around n132–138.** That spike in the first tight-face plot is the meow (jaw + interior), not a head teleport. Forehead-only traces still tick up there because the pre-meow lift and muzzle change leak a little; it is not a 30 px head cut.

### Which proposed fixes address the real cause

| Proposal | Addresses A (slideshow holds)? | Addresses B (leftover rest head)? |
|---|---|---|
| True per-frame sub-pixel motion, no dead holds | **Yes — this is the timing fix.** Replace ease-hold-ease with a continuous path (overlapping action, short *moving* holds only, residual drift, follow-through). Keep 241-periodic, start/end at 0 for the join. | No. |
| A proper ease curve | Yes, as part of the path above (slow-in/slow-out on changes of direction, not a 1 s freeze). | No. |
| RIFE / DIS frame interpolation | **No, not as the primary fix.** The holds are authored at 24 fps; interpolating them invents frames the performance did not ask for and will not remove a leftover rest-pose plate. DIS is already used at the seams and in `stabilize_mouth`. | No. |
| Subtle directional motion blur scaled to velocity | **Only after A and B are fixed**, and only where \|v\| ≳ 0.4 px/frame, shutter ~0.3–0.5 frames. On the current leftover fringe, blur would make the smear worse. Blender’s vector-blur docs also warn that 2D velocity blur cannot invent the true background behind a mover. | No; can hide stair-steps slightly once the leftover is gone. |

**Do these two things in the next round:** (1) stop revealing the painted rest-pose cat behind the warped head — inpaint or punch the cat out of the u-ex underlay and warp a gate that covers both old and new silhouettes; (2) replace dead holds with continuous, eased, overlapping head motion.

---

## 2. Tail punching a hole in the plant

### Cause

The leaves are a **static, solid u-ex occlusion card**, laid on top of the tail, with no contact deflection.

In `build_v41ae_ex.py`, `u_plant_matte()` builds one filled silhouette from u-ex leaf shapes (morphological close, contour fill, 1 px dilate, blur). `compose_layered` then: (1) composites the full cat/tail over the plate, (2) **replaces every pixel in that static matte with the unwarped u-ex leaves**. The gen clip’s own plant, and any leaf motion the I2V clip invented, are discarded. There is no per-leaf layer order: the whole filled card is in front.

That is why the tail reads as punching a hole in the plant. The tail is simply cut out wherever the card says “leaf,” and the leaf does not rotate, yield, or spring back. Gaps that the filled matte still leaves show cream tail in the middle of the foliage mass, which looks like a hole in the leaf, not a tail sliding *between* leaves.

This is distinct from the earlier v41ad-ex *speckle* bug (a per-frame green gate that classified dark tail fur as leaf and punched noisy holes in the tail). That was fixed by dropping the color gate. The remaining failure is the opposite and deeper: a solid static card with no interaction.

### Evidence

- Median optical flow on leaf-colored pixels in cycle 0 is **0.0003 px/frame** (essentially zero). Leaves do not move. Plot: `plots/tail_plant_cycle0.png`. Table: `tail_plant_metrics.csv`.
- `tail/ae-n080-cross3x.png`, `tail/ae-n189-cross3x.png` — the large mid leaf is a frozen card; the cream tail is sliced by a hard leaf edge and does not push the leaf aside.
- `tail/ae-n080-cream-in-n14leaf-3x.png` — cream tail sitting in a region that was leaf at n14 (contact without deflection). Some of this metric is also color-gate flicker; trust the 3× stills more than the raw pixel counts.
- `tail/ae-vs-gen2-n190-3x.png` — gen2 still has the tail more visible among the foliage; the ae composite has replaced that with the static u-ex card. Difference is concentrated on the plant, not the tail body.
- `tail/un0-leaf-components-labeled.png` and `tail/un0-leaf-components-labeled-3x-cross.png` — a connected-component split of the u-ex leaf shapes, used below as the segmentation plan.
- `tail/ae-n014-vs-un0-tail.png` — the composite plant matches the static plate, not a moving gen plant.

---

## 3. Organic physics — research (cited) and how to fake it in this 2D composite

### Follow-through, overlapping action, secondary action

Frank Thomas and Ollie Johnston’s principles, paraphrased from *The Illusion of Life* (pp. 47–69) at [Frank & Ollie: Physical Animation](https://frankandollie.com/PhysicalAnimation.html):

- **Follow-through:** “When the main body of the character stops all other parts continue to catch up… such as arms, long hair, clothing… floppy ears or a long tail (these follow the path of action). Nothing stops all at once.”
- **Overlapping action / drag:** when the character changes direction, appendages continue forward and catch up a number of frames later. “Long hair and animal tail will also be handled in the same manner.”
- **Slow in / slow out:** more drawings near the start and end of a move, fewer in the middle.
- **Arcs:** head turns and animal action should follow an arc, not a linear slide.
- **Secondary action:** supporting motion (breath, ear twitch, a head bob that rides the shoulders) must not distract from the main action.

[Wikipedia, “Follow through and overlapping action”](https://en.wikipedia.org/wiki/Follow_through_and_overlapping_action) records Walt Disney telling the animators, “Things don't come to a stop all at once guys; first there's one part and then another,” and lists Thomas and Johnston’s five applications, including loose flesh that *drags*, and the **moving hold**: a character may halt, but the fleshy parts continue briefly so the hold still has weight. That is the opposite of the dead holds in §1. Adobe’s summary of the same two principles is at [12 Principles of Animation](https://www.adobe.com/creativecloud/animation/discover/principles-of-animation.html).

### Leaves as damped springs; contact then overshoot

Wong, J. C. (2008), *Simulation of realistic leaf behavior and interaction with external forces for virtual reality landscapes*, University of Western Australia, models a leaf as segments with **damped angular springs** at the joints (Hooke-style), with a rigidity parameter that spans malleable to stiff: [PDF](https://api.research-repository.uwa.edu.au/ws/portalfiles/portal/3225807/Wong_Jason_C_2008.pdf).

Unreal’s [InteractiveFoliageActor](https://docs.unrealengine.com/udk/Three/InteractiveFoliageActor.html) is the standard VFX-practical version of the same idea: a touch impulse feeds a spring (`FoliageStiffness`, `FoliageDamping`, a quadratic stiffness clamp), and the result is a **rotation about an axis** used as a visual-only offset. “You can now push a bush out of the way, and it will spring back into place once you stop applying forces.” They also recommend keeping an ambient wind layer underneath the contact.

For 2D grass and reeds, [AnimateFX, “Foliage: Grass and Reeds”](https://draw.love-ink-llc.com/animatefx/BCAFXContent/BCAFX_Level-07/0704_28_Grass_and_Reeds.html) is explicit: do not treat leaves like cloth waves. “Most leaves have a supportive spine, veins that give rib-like support and a semi-rigid frame.” Overlap means the stalk moves first and the blade follows delayed; stalks flex, twist, and rock, they do not ripple.

### Ghibli / 2D notes (what the sources actually say)

This loop is meant to feel cartoonish, Ghibli-inspired, and painterly, so the leaf motion should be gentle, semi-rigid, and slightly delayed — not a cloth sim and not a big 3D bounce.

- Cátia Peres, [“Out of Gravity: Physics in animation and in the films of Hayao Miyazaki”](https://journal.animationstudies.org/article/id/91/), *Animation Studies*, discusses how Miyazaki’s timing makes even impossible physics feel credible, and how nature is a moral presence in the films. It is not a leaf-rigging manual; it is cited here only for that timing/credibility point.
- A secondary essay, [“What Makes Ghibli Look Like Ghibli?”](https://www.mlecznymlecz.com/what-makes-ghibli-look-like-ghibli-ghibli-aesthetic-explained/), observes that “Wind in Ghibli moves grass, hair, fabric, leaves, and dust at the same time, in different rhythms, in the same shot.” That is a critic’s description, not a studio document, but it matches overlapping action applied to the whole environment.

### Cat anatomy that should drive the next motion pass

- A cat’s tail has **18 to 23 caudal vertebrae** that “interlock like a chain,” which is why the motion should be a wave through the joints, not a rigid paddle: [Live Science, “Do cats communicate with their tails?”](https://www.livescience.com/animals/domestic-cats/do-cats-communicate-with-their-tails) (Reda Mohamed, Washington State University). The gen clip already supplies a useful tail flick; keep it as a vertebral wave and let the *leaves* be the ones that yield.
- Head motion should bob *from the shoulders* (the chest breath already exists, with `HEAD_LAG = 8`). Ears and whiskers should drag and settle after the skull, not move as one rigid mesh with the head.
- Ear posture is its own signal (erect vs flattened): [“Heads and Tails…”, *Animals* 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8469685/).

### How to fake this in 2D compositing (this show)

1. **Segment individual leaves** from the *static u-ex plate* (not a per-frame color gate). Treat each as a rigid card that **rotates about a stem pivot**.
2. **Layer order:** back leaves → tail/cat → front leaves → a soft contact shadow on the front leaves. The tail must *slide between* leaves, not be cut out of a single card.
3. **Drive** each front leaf with a damped angular spring. The external torque is a function of the *gen-clip tail* contour (proximity and the component of tail velocity along the leaf’s normal). When the tail approaches, the leaf is pushed off the path; when the tail leaves, the spring returns the leaf with **one mild overshoot**, then settles.
4. **Soft contact shadow:** a small, blurred, low-opacity darken under the tail on the front leaf, scaled by proximity. This is a 2D cheat for contact, not a physics light.
5. Do not simulate cloth. AnimateFX and Wong both treat leaves as semi-rigid, spine-supported blades.

Recommended spring, in per-frame units, for a gentle lo-fi / Ghibli read (derived from a standard damped oscillator so that the period and damping ratio are explicit; stiffness/damping as a *pair* follow Wong + Unreal):

- Natural period **T = 14 frames** (~0.58 s at 24 fps)
- Damping ratio **ζ = 0.60** (underdamped: one overshoot, then settle)
- `k = (2π / T)² ≈ 0.20` frame⁻²
- `c = 2 ζ √k ≈ 0.54` frame⁻¹
- Integrate: `ω += (−k · θ − c · ω + τ_contact) · dt` with `dt = 1`, then `θ += ω`
- Clamp `|θ| ≤ 12°` on the big mid leaf, `|θ| ≤ 8°` on smaller ones
- Contact torque: `τ = Kτ · s · (1 − d / r) · v_n` for d < r, with influence radius `r = 28 px`, `Kτ ≈ 0.08`, `s` a side sign so the leaf is pushed *away* from the tail’s path
- Settle in about 16–20 frames after contact; 241-periodic (run two cycles of warmup, then take one)

---

## 4. Mouth smear / lag after it opens

This is a separate failure from the head leftover in §1. It is visible on both meows, in both v41ae-ex and v41af-ex (the mouth path is the same). Structure is at its best *at the opening*; the damage is on the way down.

### Big meow (n432, local 191, gen2) — primary case

**What the 3× frames show.** At n432 the muzzle is a readable open mouth (fangs + tongue), already shallower than the gen2 source. After that the mouth does not close with the meow: it *holds*, then the opening even *widens*, fangs soften, and the lower lip drifts down, until a late close around n456–460. That is the “smears or trails behind” read.

Stills: `mouth/ae-n432-mouth3x.png`, `mouth/ae-n440-mouth3x.png`, `mouth/ae-n444-mouth3x.png`, `mouth/ae-n448-mouth3x.png`. Close strip: `mouth/ae-meow2-close-n432-454-3x.png` (af: `mouth/af-meow2-close-n432-454-3x.png`). Side-by-sides vs gen2 and vs the n188 lock: `mouth/ae-n432-vs-gen2src-vs-lock188.png`, `mouth/ae-n444-vs-gen2src-vs-lock188.png`.

**Band measurements** (longest dark/pink run in `x=700–750`; `mouth_band_metrics.csv`, `plots/mouth_band_ae_meow2.png`; af is essentially identical, `plots/mouth_band_af_meow2.png`):

| n | local | What the code is doing | Encoded opening | Lower lip y | Interior mid y |
|---|---|---|---|---|---|
| 432 | 191 | peak; stabilize to gen2 n188; src hold 188–189 | 11 px | 402 | 396.5 |
| 437 | 196 | end of src hold | 11 | 402 | 396.5 |
| 440 | 199 | 189→222 flow morph + still locked to n188 | 15 | 407 | 399.5 |
| 443 | 202 | morph ends at src 222 | 19 | 408 | 398.5 |
| 448 | 207 | post-morph; stabilize *still on* until local 210 | 18 | 413 | 404 |
| 450 | 209 | still locked | 21 | 415 | 404.5 |
| 452 | 211 | stabilize off | 17 | 414 | 405.5 |
| 456 | 215 | natural close | 7 | 412 | 408.5 |
| 460 | 219 | almost shut | 1 | 412 | 411.5 |

At the peak, gen2 n188 itself is **19 px** in the same band; the encode is **11 px**. That is the `JAW_KEEP = 0.60` chin hinge (11/19 ≈ 0.58).

During the morph (n438–443) the lower lip drops 6 px (402→408) while the interior mid moves only ~2 px (396.5→398.5). The interior lags the lips. After the morph, while stabilize is still pasting the n188 peak interior (until n451), **encoded opening stays open and even grows (15→21 px)** even though the *real* retimed gen2 source is already closing (n222 is 26 px, n225 is 17 px, n228 is 8 px). The mouth only starts a clean close once `stabilize_mouth` turns off.

### Why — three mechanisms, in order of blame

**1. `stabilize_mouth` flow-locks the interior to gen2 n188 for local 175–210 (n416–451), including the whole close after the peak.**  
The function warps the n188 reference into the *current* opening and pastes fangs/tongue there (`build_v41ae_ex.py`). During the 189→222 morph it only begins fading toward the close frame after `morph_w > 0.55` (about local 199). Then, from local 202 to 210, the gen sample is already on the close path, *but the interior is still locked to the peak*. The lips can move; the inside trails them. This matches the caller’s first suspect, and the numbers confirm it.

**2. The 189→222 flow morph does not close the mouth.**  
`GEN2_CLOSE_SRC = 222` is documented as “~60–70% opening on the way down.” In the actual gen2 clip it is **26 px — as open as, or more open than, n188 (19 px) and n189 (22 px).** The real hiss (the wide jaw the morph exists to skip) is gen2 n200–210 at **46–48 px**. Plot: `plots/gen2_source_opening_band.png`.

So the morph interpolates two *open* poses and cannot produce a close. Combined with the n188 paste, this is a recipe for smeared fangs (flow crossing, locked interior, lips starting to drift). Note: the dashed “retimed gen” line on `plots/mouth_band_ae_meow2.png` in the orange morph window is the *placeholder* `src_time_gen2` that walks through the hiss; `process()` replaces that span with the 189→222 morph. Do not read the 46 px dashed peak as something the encode failed to follow.

**3. The 60% chin hinge shallows the open mouth and can desync jaw flesh from the interior.**  
`apply_chin_hinge` squashes everything under `HINGE_Y = 388` so the lip lands at `CHIN_REST + 0.60 · drop`, with strength peaked at the meow and held for 18 frames before / 22 after. That is why n432 is 11 px against a 19 px source (and why n138 is already a much smaller mouth than gen1 n132 — see `mouth/ae-n138-vs-gen1src.png`). The hinge is applied *after* the n188 paste, so it also recompresses a locked peak interior as the jaw weight `k` decays — another way for inside and lips to drift apart.

### Small meow (n138, gen1) — related, not the same lock

There is **no** `stabilize_mouth` on the gen1 path. The 3× close strip `mouth/ae-meow1-close-n138-160-3x.png` (and af: `mouth/af-meow1-close-n138-160-3x.png`) shows a decent (already clamped) opening at n138 and an almost shut mouth by **n142 — about four frames**. `mouth/ae-n146-vs-gen1src.png` is closed, in agreement with gen1. The smear here is the *abrupt close*: `lerp_frame` (a linear dissolve between adjacent gen frames) across a fast open→shut, plus the same 60% hinge. The center-band detector under-counts this shallower mouth (fur luma on this dim plate is ~70, so a `fur−35` threshold misses a small opening that is obvious at 3×). Trust the stills, not the band height, for n138.

Temporal matte Gaussian (`temporal_gauss` on the cat alpha, σ 0.7–0.85) can slightly soften the hole in the matte, but it is not what holds the big-meow interior open until n451. The hinge warp lagging the gen motion is real as a *shallowness / desync* contributor, not the main “held open then smeared” mechanism on n432.

### Fixes for the next round (mouth only)

1. **Stop locking the interior to gen2 n188 after the peak.** If a clean-fang lock is still needed, limit it to the *approach and the peak hold* (local 175–196). From local 197 on, the interior must travel with the current gen sample (fangs, tongue, and lips from the same frame).
2. **Do not morph 189→222 as a “close.”** n222 is still fully open. Skip only the hiss (gen2 ~190–221) by morphing (or flow-lerping) **189 → a true mid-close**, e.g. gen2 **n227–228** (band opening 15→8 px), then lerp naturally to n240. Alternatively morph 189→230 (2 px, almost shut) if a faster close reads better against the audio.
3. **Shorten or drop the 191–196 source hold** if the mouth should start closing as the meow decays, instead of sitting on n188–189 for 6 frames after the peak.
4. **Raise `JAW_KEEP`** (0.85–1.0) or restrict the hinge to chin *flesh* below the lower lip, not the cavity, fangs, or tongue. The current 0.60 clamp is why the open structure is already smaller and slightly mushier than the gen source at the peak.
5. On the small meow, **do not `lerp_frame`-dissolve** across the 4-frame shut. Use a short flow morph, or hold integer source frames, and ease the close over more frames so it matches the meow decay rather than snapping.

---

## 5. Plan for the next render round (all four items)

Work on a new name (do not overwrite v41ae-ex / v41af-ex / v41ag-ex). Keep audio, scrolling title, snow, and steam unchanged. No 30-minute render. Reuse the v41z-ex / v41ae-ex meow sync (peaks at n138 and n432) unless §4’s close-path retiming slightly changes the *decay*, which should still leave the peak frames pinned.

### Step 1 — Kill the leftover rest-pose head (smear B)

In `compose_warped` (and the underlay used for `bg`):

- Build a **cat-free u-ex underlay** once: inpaint or heal the painted cat out of `u-n0` / the u-ex span in the head, chest, and ear regions, leaving window, wall, and plants. Never use a plate that still contains the rest-pose head as the fill behind a moving head.
- Warp **both** the composited cat and its alpha. Set the warp gate to the **union of the unwarped silhouette and the predicted moved silhouette**, dilated by a few pixels more than the max displacement (or warp the gate with the same field). Do not zero `dx, dy` at the destination of the move.
- Keep the pane lock, but only on pixels that are not cat at either end of the warp.
- Verify at 6× on the left ear and the left eye highlight over ease 54–70: a single sharp silhouette, no muddy fringe at the n14 position.

### Step 2 — Continuous, overlapping head motion (smear A)

- Delete the long dead holds. Author a 241-periodic head path that starts and ends at 0, moves on an arc, and uses slow-in/slow-out at direction changes.
- Allowed “holds” are **moving holds** only: residual sub-pixel drift, lagged breath (`HEAD_LAG = 8` can stay), and ear/whisker follow-through that settles over ~6–10 frames after the skull.
- Keep amplitudes modest (Ghibli-gentle), but avoid a 1–3 second freeze. A useful envelope: never let forehead |step| stay below 0.05 px/frame for more than ~12 frames unless it is the last 8 seam frames.
- Shoulders lead, head follows (already partly there via breath lag). Ears use the existing twitch rotators as *drag*, not as a rigid head weight.
- Optional, after Step 1 is clean: directional motion blur with shutter 0.3–0.5 frames, only where velocity > 0.4 px/frame, from the warp field itself (not a RIFE pass). See [Blender Vector Blur](https://docs.blender.org/manual/en/3.6/compositing/types/filter/vector_blur.html) and [Kokaram 2020](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-cvi.2019.0814) (true motion blur is an average along motion, not a frame hold).

### Step 3 — Segment the plant and put the tail *between* leaves

From `tail/un0-leaf-components-labeled.png` (component ids as in that file):

| id | Role | Stem pivot (use the pot-ward one, not the free tip) | Max θ |
|---|---|---|---|
| **7** | **Front, primary contact.** Large mid leaf the tail crosses. | **(1184, 654)** — `stem_pivot_near_pot`. Do *not* use the left tip (1041, 695). | 12° |
| **3** | **Front, secondary.** Upper leaf near the tail tip. | (1171, 577) | 8° |
| 10 | **Behind the tail** (lower). Tail slides in front of it. | (1218, 693) | 0° (static, or a tiny 3° ambient only) |
| 2, 4, 5, 9 | Right-hand pot foliage, mostly **behind or off the tail path**. Leave static, or a very small shared ambient sway. | pot-ward pivots in `summary.json` → `leaf_components` | 0–4° |

Layer order, per pixel: **(2, 4, 5, 9, 10)** → **cat + tail** (gen, with the current solid-tail inpaint) → **rotated 7 and 3** → soft contact shadow of the tail on 7 and 3.

Implementation sketch:

- Store each front leaf as an RGBA card cropped from u-ex n0 (or a quiet plate frame), with a binary matte from the filled component, feathered 1–2 px.
- Rotate each card about its stem with `cv2.warpAffine`.
- Drive θ with the damped spring in §3, using the **gen-clip tail contour** (cream tail matte from the current gen sample, *before* the static plant overlay) as the collider. A signed distance from the leaf blade to the tail, plus the tail’s per-frame velocity dotted with the leaf normal, is enough.
- Do not bring back a per-frame green color gate on the tail.
- Add a soft contact shadow: dilate the tail-over-leaf overlap 8–12 px, Gaussian blur σ ≈ 4, multiply the front leaf by ~0.85–0.9 at the contact, faded by distance.

QC: 3× tail strips at n60–92 and n170–210. The mid leaf (id 7) must visibly yield and recover; the tail must remain a solid limb that disappears *between* leaves, not through a hole in a card. Leaf flow on id 7 will no longer be zero during a flick.

### Step 4 — Mouth close path (after it opens)

- Pin the peaks as today (gen1 src 132 @ n138; gen2 src 188 @ n432).
- `stabilize_mouth` to n188: **on for local 175–196 only** (approach + peak hold). Off from local 197.
- Replace the 189→222 morph with a morph or flow-lerp **189 → 227 or 228** (true mid-close). Then lerp to 240. Still never sample the hiss (~190–221).
- Shorten the 191–196 hold to 2–3 frames, or start the close immediately after n432 if the audio decay wants it.
- `JAW_KEEP`: raise to **0.85–1.0**, or apply the hinge only below the detected lower lip so fangs and tongue keep their gen proportions.
- Small meow: replace `lerp_frame` across the n138–142 shut with a short flow morph and a slightly slower ease (on the order of 8–12 frames, not 4), still peaking at n138.
- QC at 3×: `n138–160` and `n432–460` (ae and whatever the new encode is). At n432 the opening may be closer to the 19 px source. After n432, opening height and lower-lip y must *fall* with the gen close, and the interior mid must move in step with the lip (no 6 px lip drop with a 2 px interior, and no 11→21 px re-opening while stabilize is on). Fangs stay sharp, no trailing double.

### Step 5 — Review before showing

Independent 3× pass on: ease 54–70 left ear (no leftover fringe); hold B (some residual life, not a 28-frame freeze); tail at the two flicks (leaf 7 yields, no card cutout); both meow closes (interior and lips together). State anything that still fails honestly.

---

## Evidence index

| Path | What it is |
|---|---|
| `plots/head_displacement_cycle0.png` | Knot vs measured face shift; per-frame step; MAD; first (over-firing) blend test |
| `plots/head_forehead_ear_cycle0.png` | Mouth-excluded forehead / ear remeasure |
| `plots/head_displacement_all600.png` | All 600 frames, both cycle joins |
| `plots/tail_plant_cycle0.png` | Leaf IoU, leaf flow ≈ 0, cream/leaf occupancy |
| `plots/mouth_band_ae_meow2.png` / `_ae_meow1.png` / `_af_*.png` | Band opening and lip vs interior (ignore dashed gen during the orange morph window) |
| `plots/gen2_source_opening_band.png` | Why n222 is not a close |
| `head/` | 3× / 6× face, ear, overlays, consecutive diffs |
| `tail/` | Plant stills, ae vs gen2, labeled leaf components |
| `mouth/` | 3× muzzle stills and gen/lock comparisons |
| `head_displacement.csv`, `head_forehead_ear.csv`, `tail_plant_metrics.csv`, `mouth_band_metrics.csv`, `summary.json`, `mouth_head_summary.json` | Raw numbers |

