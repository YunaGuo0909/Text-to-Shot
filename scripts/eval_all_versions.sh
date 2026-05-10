#!/bin/bash
# Evaluate all available FM checkpoints for comparison table
export PYTHONPATH=.

EVAL="python experiments/flow_matching/evaluate.py"

echo "=========================================="
echo "Evaluating all FM versions"
echo "=========================================="

# v4: FM E.T. only (58k, person_dim=3) - baseline without AMASS
if [ -f /transfer/fm-v4-checkpoints/fm_final.pth ]; then
    echo ""
    echo ">>> FM v4 (E.T. only 58k, no AMASS)"
    $EVAL --checkpoint /transfer/fm-v4-checkpoints/fm_final.pth --device cuda --max-samples 1000
fi

# v5: FM + AMASS (326k, person_dim=3)
if [ -f /transfer/fm-v5-checkpoints/fm_final.pth ]; then
    echo ""
    echo ">>> FM v5 (E.T. + AMASS 326k)"
    $EVAL --checkpoint /transfer/fm-v5-checkpoints/fm_final.pth --device cuda --max-samples 1000
fi

# v6: FM + smooth loss (660k, person_dim=3)
if [ -f /transfer/fm-v6-checkpoints/fm_final.pth ]; then
    echo ""
    echo ">>> FM v6 (merged-v7 660k)"
    $EVAL --checkpoint /transfer/fm-v6-checkpoints/fm_final.pth --device cuda --max-samples 1000
fi

# v7: FM + raw yaw (660k, person_dim=4, expected: mode collapse)
if [ -f /transfer/fm-v7-checkpoints/fm_final.pth ]; then
    echo ""
    echo ">>> FM v7 (raw yaw, expected mode collapse)"
    $EVAL --checkpoint /transfer/fm-v7-checkpoints/fm_final.pth --device cuda --max-samples 1000
fi

# v8: FM + sin/cos yaw (660k, person_dim=5)
if [ -f /transfer/fm-v8-checkpoints/fm_final.pth ]; then
    echo ""
    echo ">>> FM v8 (sin/cos yaw)"
    $EVAL --checkpoint /transfer/fm-v8-checkpoints/fm_final.pth --device cuda --max-samples 1000
fi

# v9: skip for now, training in progress
# Run separately after training completes:
# $EVAL --checkpoint /transfer/fm-v9-checkpoints/fm_final.pth --device cuda --max-samples 1000

echo ""
echo "=========================================="
echo "All evaluations complete"
echo "=========================================="
