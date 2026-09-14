"""Born vs multislice loss curves for the paper (no stochastic curve).

Reads the 100-iteration tBL-WSe2 runs under demo/output/test_100/ and writes
electron_loss_comparison.png into the paper directory. Derived from
loss_plots.py; the parsing is identical, only the inputs and output differ.
"""

import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/home/dnz75396/ptyrad/demo/output/test_100"
RUN = ("20260731_full_N16384_dp128_flipT100_random32_p6_1obj_12slice_dz1_"
       "plr1e-4_oalr5e-4_oplr5e-4_orblur0.5_ozblur1.0_mamp0.03_4.0_"
       "oathr0.98_oposc_sng1.0_spr0.1")
FILE_PATHS = {
    "ISS": os.path.join(BASE, "tBL_WSe2_born", RUN, "loss_results.txt"),
    "multislice": os.path.join(BASE, "tBL_WSe2_multislice", RUN, "loss_results.txt"),
}
OUT = "/home/dnz75396/ptypy/draft_paper/electron_loss_comparison.png"

PATTERN = re.compile(
    r"Iter:\s*(\d+),\s*Total Loss:\s*([\d.]+).*?in\s*([\d.]+)\s*sec",
    re.IGNORECASE,
)


def parse_log_file(filepath):
    iterations, losses, times = [], [], []
    with open(filepath) as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                iterations.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                times.append(float(m.group(3)))
    cumulative, total = [], 0.0
    for t in times:
        total += t
        cumulative.append(total)
    return iterations, losses, cumulative


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"ISS": "tab:blue", "multislice": "tab:green"}
    for label, path in FILE_PATHS.items():
        it, loss, cum = parse_log_file(path)
        n = len(it)
        mean_t = cum[-1] / n
        print(f"{label}: {n} iters, mean {mean_t:.2f} s/iter, "
              f"final loss {loss[-1]:.4f}")
        ax1.semilogy(it, loss, color=colors[label], lw=1.5, label=label)
        ax2.semilogy(cum, loss, color=colors[label], lw=1.5, label=label)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("total loss")
    ax2.set_xlabel("cumulative wall time (s)")
    ax2.set_ylabel("total loss")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT, dpi=200)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
