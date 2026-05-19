#!/bin/zsh
# Generate same dolly-in prompt across all versions for cross-version comparison
# Usage: zsh scripts/gen_cross_version.sh

PROMPT="As the character moves forward, the camera pushes in"
MOTION="dolly-in"
SHOT="medium-shot"

versions=(v4 v5 v6 v7 v8 v9 v10)
ckpts=(
    "/transfer/fm-v4-checkpoints/fm_final.pth"
    "/transfer/fm-v5-checkpoints/fm_final.pth"
    "/transfer/fm-v6-checkpoints/fm_final.pth"
    "/transfer/fm-v7-checkpoints/fm_final.pth"
    "/transfer/fm-v8-checkpoints/fm_final.pth"
    "/transfer/fm-v10-checkpoints/fm_best.pth"
    "/transfer/fm-v11-checkpoints/fm_best.pth"
)

outdir="/transfer/fm-v10-outputs/cross_version"
mkdir -p "$outdir"

for i in {1..7}; do
    ver=${versions[$i]}
    ckpt=${ckpts[$i]}
    echo "=== $ver: $ckpt ==="
    if [ ! -f "$ckpt" ]; then
        echo "  SKIP: checkpoint not found"
        continue
    fi
    PYTHONPATH=. python experiments/flow_matching/generate.py \
        --checkpoint "$ckpt" \
        --text "$PROMPT" \
        --motion "$MOTION" \
        --shot-type "$SHOT" \
        --lookat \
        --output-dir "$outdir"
    # rename output to include version tag
    latest=$(ls -t "$outdir"/fm_joint_*.png 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        mv "$latest" "$outdir/fm_joint_${ver}_${MOTION}.png"
        echo "  Saved: $outdir/fm_joint_${ver}_${MOTION}.png"
    fi
    echo ""
done

echo "=== Done ==="
echo "Outputs in $outdir/"
ls "$outdir"/fm_joint_*.png 2>/dev/null
