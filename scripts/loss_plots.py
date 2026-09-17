import os
import re
import matplotlib.pyplot as plt

# File paths mapped to their display labels for the plot legend
FILE_PATHS = {
    "Born": "/home/dnz75396/ptyrad/demo/output/test_40/tBL_WSe2_born/20260730_full_N16384_dp128_flipT100_random32_p6_1obj_12slice_dz1_plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_oathr0.98_oposc_sng1.0_spr0.1/total_loss_born.txt",
    "Multislice": "/home/dnz75396/ptyrad/demo/output/test_40/tBL_WSe2_multislice/20260730_full_N16384_dp128_flipT100_random32_p6_1obj_12slice_dz1_plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_oathr0.98_oposc_sng1.0_spr0.1/total_loss_multislice.txt",
    "Stochastic": "/home/dnz75396/ptyrad/demo/output/tBL_WSe2_stochastic_v0/block3/20260730_full_N16384_dp128_flipT100_random32_p6_1obj_12slice_dz1_plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_oathr0.98_oposc_sng1.0_spr0.1/total_loss.txt",
}

# Regex pattern to match Iteration, Total Loss, and time (in seconds)
# Works whether or not there are leading timestamp prefix details
PATTERN = re.compile(
    r"Iter:\s*(\d+),\s*Total Loss:\s*([\d.]+).*?in\s*([\d.]+)\s*sec",
    re.IGNORECASE,
)


def parse_log_file(filepath):
    iterations = []
    total_losses = []
    step_times = []

    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    with open(filepath, "r") as f:
        for line in f:
            match = PATTERN.search(line)
            if match:
                iteration = int(match.group(1))
                loss = float(match.group(2))
                sec = float(match.group(3))

                iterations.append(iteration)
                total_losses.append(loss)
                step_times.append(sec)

    # Compute cumulative time in seconds
    cumulative_time = []
    current_total = 0.0
    for t in step_times:
        current_total += t
        cumulative_time.append(current_total)

    return {
        "iterations": iterations,
        "total_losses": total_losses,
        "cumulative_time": cumulative_time,
    }


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, path in FILE_PATHS.items():
        data = parse_log_file(path)
        if data is None or not data["iterations"]:
            continue

        # Plot 1: Total Loss vs. Iteration
        ax1.plot(
            data["iterations"],
            data["total_losses"],
            marker="o",
            markersize=3,
            label=label,
        )

        # Plot 2: Total Loss vs. Total Time (Seconds)
        ax2.plot(
            data["cumulative_time"],
            data["total_losses"],
            marker="o",
            markersize=3,
            label=label,
        )

    # Styling Plot 1
    ax1.set_title("Total Loss vs. Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Total Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Styling Plot 2
    ax2.set_title("Total Loss vs. Total Time")
    ax2.set_xlabel("Cumulative Time (seconds)")
    ax2.set_ylabel("Total Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    # Save plot as PNG and show window
    output_filename = "loss_comparison_plots.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plots successfully saved to {output_filename}")
    plt.show()


if __name__ == "__main__":
    main()
