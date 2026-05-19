#!/bin/bash
# Batch generation for FM v11 — all 9 motion types, 3 passes:
#   pass1: full pipeline (smooth + enforce-constraints)
#   pass2: raw (no smooth, no lookat)
#   pass3: raw + lookat (geometry-correct orientation, no smoothing)
#
# Usage:
#   bash scripts/gen_v11_all.sh
#   bash scripts/gen_v11_all.sh --pass1-only
#   bash scripts/gen_v11_all.sh --guidance 5.0
#
# Output: /transfer/fm-v11-outputs/{full,raw,lookat}/

set -e

CKPT="/transfer/fm-v11-checkpoints/fm_final.pth"
OUT_BASE="/transfer/fm-v11-outputs"
OUT_FULL="$OUT_BASE/full"
OUT_RAW="$OUT_BASE/raw"
OUT_LOOKAT="$OUT_BASE/lookat"

GUIDANCE=3.0
STEPS=50
PASS1=true
PASS2=true
PASS3=true

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --pass1-only) PASS2=false; PASS3=false ;;
        --guidance)   GUIDANCE="$2"; shift ;;
        --steps)      STEPS="$2"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

MOTIONS=(static dolly-in dolly-out pan-left pan-right crane-up crane-down track orbit)

declare -A PROMPTS
PROMPTS[static]="The camera remains static while the character walks forward"
PROMPTS[dolly-in]="As the character moves forward, the camera pushes in toward them"
PROMPTS[dolly-out]="The camera pulls back as the character stands still"
PROMPTS[pan-left]="The camera pans left as the character walks forward"
PROMPTS[pan-right]="The camera pans right as the character stands still"
PROMPTS[crane-up]="The camera cranes up as the character moves forward"
PROMPTS[crane-down]="The camera lowers down while the character stands still"
PROMPTS[track]="The camera tracks the character as they walk to the right"
PROMPTS[orbit]="The camera orbits around the character as they stand still"

mkdir -p "$OUT_FULL" "$OUT_RAW" "$OUT_LOOKAT"

echo "========================================"
echo " FM v11 Batch Generation"
echo "========================================"
echo " checkpoint:    $CKPT"
echo " guidance:      $GUIDANCE"
echo " steps:         $STEPS"
echo " output base:   $OUT_BASE"
echo "========================================"

# -----------------------------------------------
# PASS 1: Full pipeline (smooth + constraints)
# -----------------------------------------------
if [ "$PASS1" = true ]; then
    echo ""
    echo "=== PASS 1: Full pipeline (smooth + enforce-constraints) ==="
    for motion in "${MOTIONS[@]}"; do
        echo -n "  [$motion] ... "
        PYTHONPATH=. python experiments/flow_matching/generate.py \
            --checkpoint "$CKPT" \
            --text "${PROMPTS[$motion]}" \
            --motion "$motion" \
            --guidance-scale "$GUIDANCE" \
            --steps "$STEPS" \
            --enforce-constraints \
            --output-dir "$OUT_FULL" \
            2>&1 | tail -1
    done
    echo "  -> $OUT_FULL/"
fi

# -----------------------------------------------
# PASS 2: Raw (no smoothing, no lookat)
# -----------------------------------------------
if [ "$PASS2" = true ]; then
    echo ""
    echo "=== PASS 2: Raw output (no smooth, no lookat) ==="
    for motion in "${MOTIONS[@]}"; do
        echo -n "  [$motion] ... "
        PYTHONPATH=. python experiments/flow_matching/generate.py \
            --checkpoint "$CKPT" \
            --text "${PROMPTS[$motion]}" \
            --motion "$motion" \
            --guidance-scale "$GUIDANCE" \
            --steps "$STEPS" \
            --no-smooth \
            --output-dir "$OUT_RAW" \
            2>&1 | tail -1
    done
    echo "  -> $OUT_RAW/"
fi

# -----------------------------------------------
# PASS 3: Raw + look-at
# -----------------------------------------------
if [ "$PASS3" = true ]; then
    echo ""
    echo "=== PASS 3: Raw + look-at (no smooth, geometry orientation) ==="
    for motion in "${MOTIONS[@]}"; do
        echo -n "  [$motion] ... "
        PYTHONPATH=. python experiments/flow_matching/generate.py \
            --checkpoint "$CKPT" \
            --text "${PROMPTS[$motion]}" \
            --motion "$motion" \
            --guidance-scale "$GUIDANCE" \
            --steps "$STEPS" \
            --no-smooth \
            --lookat \
            --output-dir "$OUT_LOOKAT" \
            2>&1 | tail -1
    done
    echo "  -> $OUT_LOOKAT/"
fi

# -----------------------------------------------
# Summary stats
# -----------------------------------------------
echo ""
echo "========================================"
echo " Summary stats (PASS 1 full outputs)"
echo "========================================"
python3 - <<'PYEOF'
import numpy as np, glob, os

out_dir = "/transfer/fm-v11-outputs/full"
person_files = sorted(glob.glob(os.path.join(out_dir, "fm_person_*.npy")))
camera_files = sorted(glob.glob(os.path.join(out_dir, "fm_camera_*.npy")))

if not person_files:
    print("  No outputs found in", out_dir)
else:
    print(f"  {'motion':<12}  {'person_disp':>11}  {'cam_dist_range':>15}  {'cam_jerk':>9}")
    print(f"  {'-'*12}  {'-'*11}  {'-'*15}  {'-'*9}")
    for pf, cf in zip(person_files, camera_files):
        tag = os.path.basename(pf).replace("fm_person_", "").replace(".npy", "")
        motion = tag.split("_")[0]
        p = np.load(pf)
        c = np.load(cf)
        disp = float(np.linalg.norm(p[-1, :3] - p[0, :3]))
        dists = np.linalg.norm(c[:, :3] - p[:, :3], axis=1)
        cam_jerk = float(np.mean(np.linalg.norm(np.diff(np.diff(np.diff(c[:, :3], axis=0), axis=0), axis=0), axis=1))) if c.shape[0] >= 4 else 0.0
        dist_range = f"{dists.min():.2f}->{dists.max():.2f}m"
        print(f"  {motion:<12}  {disp:>11.3f}m  {dist_range:>15}  {cam_jerk:>9.6f}")
PYEOF

echo ""
echo "All done."
