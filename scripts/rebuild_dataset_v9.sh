#!/bin/bash
# Rebuild dataset for v9 training with all fixes applied:
# 1. E.T.: trajectory-based labels (not caption), jerk filtering, no look-at proxy
# 2. AMASS: track skipped for stationary people
# 3. HumanML3D: same track fix via shared generate_camera_for_person
# 4. Merge with only real person data
# 5. Compute norm_stats with person_dim=3

set -e

echo "=== Step 1: Re-preprocess E.T. (trajectory-based labels + jerk filter) ==="
PYTHONPATH=. python scripts/preprocess_et_data.py \
    --et-root /transfer/et-data \
    --output-root /transfer/stc-data-v9b \
    --require-person \
    --num-frames 48

echo ""
echo "=== Step 2: Re-prepare AMASS (track fix) ==="
PYTHONPATH=. python scripts/prepare_amass.py \
    --amass-root /transfer/amassdata \
    --output-root /transfer/amass-stc-data-v9bb \
    --num-frames 48

echo ""
echo "=== Step 3: Re-prepare HumanML3D (track fix) ==="
PYTHONPATH=. python scripts/prepare_humanml3d.py \
    --amass-root /transfer/amassdata \
    --humanml3d-root /transfer/HumanML3D \
    --output-root /transfer/humanml3d-stc-data-v9bb \
    --num-frames 48

echo ""
echo "=== Step 4: Merge all sources ==="
PYTHONPATH=. python scripts/merge_datasets.py \
    --sources /transfer/stc-data-v9b /transfer/amass-stc-data-v9bb /transfer/humanml3d-stc-data-v9bb \
    --output-root /transfer/merged-v9b \
    --person-dim 3 \
    --camera-dim 6

echo ""
echo "=== Step 5: Compute norm stats (person_dim=3) ==="
python compute_norm_stats.py 3 /transfer/merged-v9b

echo ""
echo "=== Step 6: Verify with diagnostic ==="
python scripts/diagnose_v6_issues.py --data-root /transfer/merged-v9b --max-samples 2000

echo ""
echo "=== DONE ==="
echo "Update v9.yaml data_root to /transfer/merged-v9b"
echo "Update v9.yaml norm_stats_path to /transfer/merged-v9b/norm_stats_v9.json"
