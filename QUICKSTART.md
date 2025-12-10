# Quick Start Guide

## System Status
✅ **All components implemented and tested successfully**  
✅ **MPS (Apple Silicon GPU) acceleration working**  
✅ **Training pipeline verified**

## Your Setup
- **Device**: Apple Silicon Mac with MPS acceleration
- **TensorFlow**: 2.20.0
- **Virtual Environment**: `.venv` (activated)

## Fixed Issues
1. ✅ MPS device detection (was showing CPU, now shows MPS)
2. ✅ dtype mismatches in channel normalization
3. ✅ Import path issues in models

## Run Training

### Test Mode (Quick Verification - 1 Epoch)
```bash
source .venv/bin/activate
python train.py --test_mode
```

### Full Training
```bash
source .venv/bin/activate
python train.py
```

### Monitor Progress
```bash
# In another terminal
source .venv/bin/activate
tensorboard --logdir ./logs
```
Then open: http://localhost:6006

## Expected Training Output
- Device: MPS (Apple Silicon GPU acceleration)
- Batch processing with progress bars
- Loss decreasing (becomes more negative as BF gain increases)  
- Beamforming gain increasing over epochs
- Checkpoints saved to `./checkpoints/`

## After Training

### Evaluate Model
```bash
python evaluate.py --checkpoint_dir ./checkpoints --output_dir ./results
```

### View Results
Results will be saved as PNG plots in `./results/`:
- `figure_6_reproduction.png` - BF gain vs SNR
- `figure_7_reproduction.png` - BF gain vs sensing steps

## All Available Commands

```bash
# Verify device
python device_setup.py

# Test individual components
python config.py
python utils.py
python channel_model.py
python models/bs_controller.py
python models/ue_controller.py
python models/beam_alignment.py
python metrics.py

# Training options
python train.py --help
python train.py --epochs 50 --batch_size 256
python train.py --lr 0.001

# Evaluation options
python evaluate.py --mode fig6
python evaluate.py --mode fig7
python evaluate.py --mode all
```

## Files Created (15 total)

**Core (7 files)**:
- `config.py` - System configuration
- `device_setup.py` - MPS/CUDA/CPU detection
- `utils.py` - Utility functions
- `channel_model.py` - mmWave channel
- `metrics.py` - Performance tracking
- `train.py` - Training script
- `evaluate.py` - Evaluation & plots

**Models (4 files)**:
- `models/__init__.py`
- `models/bs_controller.py` - BS learned codebook
- `models/ue_controller.py` - UE adaptive RNN
- `models/beam_alignment.py` - End-to-end model

**Documentation (4 files)**:
- `README.md` - Full documentation
- `requirements.txt` - Dependencies
- `QUICKSTART.md` - This file
- Project walkthrough (in artifacts)

## Troubleshooting

If you see "ModuleNotFoundError":
```bash
source .venv/bin/activate  # Make sure venv is activated
```

If MPS not detected:
```bash
python device_setup.py  # Should show "MPS detected"
```

If training fails:
```bash
python train.py --test_mode  # Run quick test first
```

---

**Ready to train!** 🚀

Start with: `python train.py --test_mode`
