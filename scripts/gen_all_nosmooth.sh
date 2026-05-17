#!/bin/bash
# Generate all 9 motion types with --no-smooth and --lookat
# to compare raw model output vs post-processed
#
# Usage: bash scripts/gen_all_nosmooth.sh

CKPT="/transfer/fm-v10-checkpoints/fm_best.pth"
OUT_RAW="outputs/raw_nosmooth"
OUT_LOOKAT="outputs/raw_lookat"

mkdir -p "$OUT_RAW" "$OUT_LOOKAT"

declare -A PROMPTS
PROMPTS[static]="The camera remains static while the character walks forward"
PROMPTS[dolly-in]="As the character moves forward, the camera pushes in"
PROMPTS[dolly-out]="The camera pulls out as the character stands still"
PROMPTS[pan-left]="The camera pans left as the character walks forward"
PROMPTS[pan-right]="The camera pans right as the character stands still"
PROMPTS[crane-up]="The camera cranes up as the character moves forward"
PROMPTS[crane-down]="The camera lowers while the character stands still"
PROMPTS[track]="The camera tracks the character as they walk to the right"
PROMPTS[orbit]="The camera orbits around the character as they stand still"

echo "=== Pass 1: Raw (no smoothing, no lookat) ==="
for motion in static dolly-in dolly-out pan-left pan-right crane-up crane-down track orbit; do
    echo "--- $motion ---"
    PYTHONPATH=. python experiments/flow_matching/generate.py \
        --checkpoint "$CKPT" \
        --text "${PROMPTS[$motion]}" \
        --motion "$motion" \
        --no-smooth \
        --output-dir "$OUT_RAW"
done

echo ""
echo "=== Pass 2: Raw + look-at (no smoothing, with lookat) ==="
for motion in static dolly-in dolly-out pan-left pan-right crane-up crane-down track orbit; do
    echo "--- $motion ---"
    PYTHONPATH=. python experiments/flow_matching/generate.py \
        --checkpoint "$CKPT" \
        --text "${PROMPTS[$motion]}" \
        --motion "$motion" \
        --no-smooth \
        --lookat \
        --output-dir "$OUT_LOOKAT"
done

echo ""
echo "=== Done ==="
echo "Raw outputs:   $OUT_RAW/"
echo "Lookat outputs: $OUT_LOOKAT/"
echo ""
echo "Compare person motion range (should show if model generates motion or not):"
echo "python -c \"
import numpy as np, glob
for f in sorted(glob.glob('$OUT_RAW/fm_person_*.npy')):
    d = np.load(f)
    rng = d.max(axis=0) - d.min(axis=0)
    disp = np.linalg.norm(d[-1] - d[0])
    print(f'{f.split(\"/\")[-1]:50s}  range={rng}  disp={disp:.4f}')
\"
"
