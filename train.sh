# Figure 4
python train.py --scheme C1 -T 16 --epochs 100
python train.py --scheme C2 -T 16 --epochs 100
python train.py --scheme C3 -T 16 --epochs 100

# Figure 5
# Train C2 models
# for T in 1 3 5 7 8 9 15; do
#     python train.py --scheme C2 -T $T --epochs 100
# done
# # Train C3 models
# for T in 1 3 5 7 8 9 15; do
#     python train.py --scheme C3 -T $T --epochs 100
# done