#!/bin/bash
set -e

echo "============================================"
echo "  ProvenanceGuard Full Experiment Suite"
echo "============================================"

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: Set ANTHROPIC_API_KEY first"
    echo "  export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi

mkdir -p results/full

# Quick validation (5 min, ~$2)
echo ""
echo "[Quick] Running small-scale validation..."
python3 scripts/full_experiment.py \
    --seed 42 --n-corpus 100 --n-queries 20 --n-adversarial 10 \
    --output-dir results/quick

# Medium run (30 min, ~$20)
echo ""
echo "[Medium] Running medium-scale experiment..."
for seed in 42 123 456; do
    echo "  Seed: $seed"
    python3 scripts/full_experiment.py \
        --seed $seed --n-corpus 500 --n-queries 100 --n-adversarial 25 \
        --output-dir results/medium
done

# Full run (3-5 hrs, ~$100) — uncomment when ready
# echo ""
# echo "[Full] Running full-scale experiment..."
# for seed in 42 123 456 789 101 202 303 404 505 606; do
#     echo "  Seed: $seed"
#     python3 scripts/full_experiment.py \
#         --seed $seed --n-corpus 5000 --n-queries 1000 --n-adversarial 250 \
#         --output-dir results/full --skip-generation
# done

echo ""
echo "============================================"
echo "  Done! Results in results/"
echo "============================================"
