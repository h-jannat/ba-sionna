#!/bin/bash
# Script to train C2 and C3 models for all T values needed for Figure 5

# T values from the paper for Figure 5
T_VALUES=(1 3 5 7 8 9 15)

echo "Training C2 and C3 models for Figure 5"
echo "T values: ${T_VALUES[@]}"
echo "==========================================="

# Train C2 models (Learnable codebook, Fixed start)
echo ""
echo "Training C2 models..."
for T in "${T_VALUES[@]}"; do
    echo "  Training C2 with T=$T..."
    python train.py --scheme C2 --num_sensing_steps $T --epochs 100
done

# Train C3 models (Learnable codebook, Random start)  
echo ""
echo "Training C3 models..."
for T in "${T_VALUES[@]}"; do
    echo "  Training C3 with T=$T..."
    python train.py --scheme C3 --num_sensing_steps $T --epochs 100
done

echo ""
echo "==========================================="
echo "Training complete!"
echo ""
echo "To generate Figure 5, run:"
echo "  python evaluate_paper_figures.py --figure 5c --num_samples 1000"
