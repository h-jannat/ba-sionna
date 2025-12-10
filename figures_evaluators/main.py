# ==================== Main Execution ====================


import argparse
import os
from config import Config
from device_setup import setup_device
from figures_evaluators.figure4 import generate_figure_4_c1_c2_c3
from figures_evaluators.figure5 import generate_figure_5_c2_c3


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Reproduce figures 4 and 5 from the paper (C1/C2/C3 comparisons)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results", help="Output directory for plots"
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="all",
        choices=["all", "4", "5"],
        help="Which figure(s) to generate: 4 (C1/C2/C3 vs SNR), 5 (C2/C3 vs T)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=2000, help="Number of samples for evaluation"
    )

    args = parser.parse_args()

    # Setup device
    setup_device(verbose=True)

    # Print config
    print("\n" + "=" * 80)
    print("PAPER FIGURE REPRODUCTION")
    print("=" * 80)
    print("\nGenerating correct figure implementations:")
    print("  - Figure 4: C1, C2, C3 comparison vs SNR")
    print("  - Figure 5: C2, C3 comparison vs T (sensing steps)")
    print("=" * 80)
    Config.print_config()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate figures
    if args.figure in ["all", "4"]:
        generate_figure_4_c1_c2_c3(Config, args.output_dir, args.num_samples)

    if args.figure in ["all", "5"]:
        generate_figure_5_c2_c3(Config, args.output_dir, args.num_samples)

    print("\n" + "=" * 80)
    print("FIGURE REPRODUCTION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nGenerated figures:")
    if args.figure in ["all", "4"]:
        print("  - figure_4_c1_c2_c3.png")
    if args.figure in ["all", "5"]:
        print("  - figure_5_c2_c3_performance_vs_T.png")


if __name__ == "__main__":
    main()
