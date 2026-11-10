# Plan: mouth close after the meow

Issue #3. This commit is the plan only. The mouth-path changes are being developed elsewhere and will be pushed to `fix/mouth-lag` later.

## What is going wrong

On the big meow (around frame 432, local frame 191, gen2) the mouth opens into a readable shape, then hangs open and smears instead of closing with the sound. The damage is on the way down. Three mechanisms, in the order they matter:

1. `stabilize_mouth` pastes the interior of gen2 frame 188 from local frames 175 through 210, which includes the whole close. The lips move and the opening grows from about 11 px to about 21 px while the fangs and tongue stay locked to the peak, so they go soft. The inside lags the lips: during the morph the lower lip drops about 6 px while the interior mid moves about 2 px.
2. The close morph runs from gen2 frame 189 to frame 222. Frame 222 is still fully open (about 26 px in the measurement band, as open as the peak). The morph never aims at a shut mouth. The hiss that morph exists to skip is about gen2 frames 190–221. A true mid-close is closer to frames 227–228 (opening falling through about 15 px to 8 px).
3. `JAW_KEEP` is 0.60. The chin hinge squashes everything under the hinge line so the open mouth is about 60% of the source (about 11 px encoded against a 19 px gen2 peak). The hinge runs after the locked paste, so it also recompresses that frozen interior as the jaw weight decays.

The small meow (around frame 138, gen1) does not use `stabilize_mouth`. It is already a decent opening at frame 138 and nearly shut by about frame 142. That smear is the abrupt close: `lerp_frame` dissolves across a four-frame shut, with the same 60% hinge.

Keep the peaks where they are: gen1 source frame 132 at output frame 138, and gen2 source frame 188 at output frame 432.

## What to change

1. Keep `stabilize_mouth` only for the approach and the peak hold (local frames 175–196). From local 197 on, fangs, tongue, and lips come from the same current gen sample.
2. Morph or flow-lerp frame 189 into gen2 frame 227 or 228, then lerp on to 240. Leave the hiss (about 190–221) unsampled. Shorten the source hold at local 191–196 to two or three frames, or start the close immediately after frame 432 if the meow decay wants that.
3. Raise `JAW_KEEP` to about 0.85–1.0, or apply the hinge only to chin flesh below the lower lip so the cavity, fangs, and tongue keep the proportions of the gen frame.
4. On the small meow, replace the `lerp_frame` dissolve across the four-frame shut with a short flow morph, and ease the close over about 8–12 frames, still peaking at frame 138.

## How to check

At 3×, frames 138–160 and 432–460. At 432 the opening can sit closer to the 19 px source. After 432 the opening height and the lower lip fall with the gen close, and the interior mid moves with the lip. Fangs stay sharp, with no trailing double and no re-opening while stabilize is on.
