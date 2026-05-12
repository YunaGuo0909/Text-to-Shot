#!/bin/bash
export PYTHONPATH=.

CKPT="/transfer/fm-v10-checkpoints/fm_best.pth"
OUT="/transfer/fm-v10-outputs"

# Without constraints
python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera remains static while the character walks forward" --motion static --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "As the character moves forward, the camera pushes in" --motion dolly-in --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera pulls out as the character stands still" --motion dolly-out --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera pans left as the character walks forward" --motion pan-left --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera pans right as the character stands still" --motion pan-right --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera cranes up as the character moves forward" --motion crane-up --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera lowers while the character stands still" --motion crane-down --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera tracks the character as they walk to the right" --motion track --guidance-scale 3.0 --output-dir "$OUT"

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera orbits around the character as they stand still" --motion orbit --guidance-scale 3.0 --output-dir "$OUT"

# With constraints (for comparison)
python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "As the character moves forward, the camera pushes in" --motion dolly-in --guidance-scale 3.0 --output-dir "${OUT}_constrained" --enforce-constraints

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera orbits around the character as they stand still" --motion orbit --guidance-scale 3.0 --output-dir "${OUT}_constrained" --enforce-constraints

python experiments/flow_matching/generate.py --checkpoint "$CKPT" --text "The camera remains static while the character walks forward" --motion static --guidance-scale 3.0 --output-dir "${OUT}_constrained" --enforce-constraints

echo "Done. Outputs in $OUT and ${OUT}_constrained"
