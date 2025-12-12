import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import tensorflow as tf

from figures_evaluators.common import evaluate_at_snr_fixed_channels, load_c3_model


def generate_figure_4_cdl_comparison(config, output_dir="./results", num_samples=2000):
    """
    Figure 4 (same axes as paper): BF gain and satisfaction probability vs SNR.

    Change vs paper: instead of comparing schemes, we compare channel variants
    (CDL-A..E) using the same trained C3 model.
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 4: C3 PERFORMANCE ACROSS CDL VARIANTS")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Keep the same SNR axis as the existing implementation
    snr_range = np.arange(-15, 26, 5)
    batch_size = config.BATCH_SIZE
    target_snr_db = float(getattr(config, "SNR_TARGET", 20.0))

    # Compare each CDL variant separately at evaluation time
    cdl_variants = list(getattr(config, "CDL_MODELS", ["A", "B", "C", "D", "E"]))

    checkpoint_dir = f"./checkpoints_C3_T{config.T}"

    results = {
        cdl: {"bf_gain": [], "sat_prob": []}
        for cdl in cdl_variants
    }

    # Evaluate one model per CDL variant (same weights, different channel condition)
    for cdl in cdl_variants:
        print(f"\nLoading C3 model for evaluation on CDL-{cdl}...")
        model = load_c3_model(config, checkpoint_dir, cdl_models=[cdl])

        # Pre-generate a fixed set of channels for this CDL, reused across all SNR points.
        # This makes the SNR trend reflect measurement noise, not resampling variance.
        fixed_channels = model.channel_model.generate_channel(int(num_samples))
        # Also fix the sweep start indices across SNR points (avoid extra randomness).
        fixed_start_idx = tf.random.uniform(
            [int(num_samples)], minval=0, maxval=int(config.NCB), dtype=tf.int32
        )

        print(f"Evaluating CDL-{cdl}...")
        for snr_db in tqdm(snr_range, desc=f"CDL-{cdl}"):
            metrics = evaluate_at_snr_fixed_channels(
                model,
                fixed_channels,
                float(snr_db),
                batch_size,
                target_snr_db,
                start_idx=fixed_start_idx,
            )
            results[cdl]["bf_gain"].append(metrics["mean_bf_gain_db"])
            results[cdl]["sat_prob"].append(metrics["satisfaction_prob"])

    # Plot (keep axes/labels structure)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for i, cdl in enumerate(cdl_variants):
        marker = markers[i % len(markers)]
        ax1.plot(
            snr_range,
            results[cdl]["bf_gain"],
            marker=marker,
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            label=f"CDL-{cdl}",
        )
        ax2.plot(
            snr_range,
            results[cdl]["sat_prob"],
            marker=marker,
            linestyle="-",
            linewidth=2.0,
            markersize=7,
            label=f"CDL-{cdl}",
        )

    ax1.set_xlabel("SNR [dB]", fontsize=14)
    ax1.set_ylabel("Beamforming gain [dB]", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12, loc="best")
    ax1.set_title("(a) Beamforming Gain vs SNR", fontsize=14)

    ax2.set_xlabel("SNR [dB]", fontsize=14)
    ax2.set_ylabel("Satisfaction probability", fontsize=14)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12, loc="best")
    ax2.set_title("(b) Satisfaction Probability vs SNR", fontsize=14)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "figure_4_cdl_comparison.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved Figure 4 to {fig_path}")
    plt.close()

    return results
