import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def style(ax):
    ax.set_facecolor("#1e1e1e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")

def plot_distribution(data, title, xlabel, analytical=None, filename="distribution.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1e1e1e")
    style(ax)
    ax.hist(data, bins=50, color="#1f77b4", alpha=0.8, edgecolor="#444444", density=True)
    if analytical is not None:
        ax.axvline(analytical, color="#ff4d4d", linewidth=2, linestyle="--", label=f"Analytical: {analytical:.4f}")
        ax.axvline(np.mean(data), color="#00cc66", linewidth=2, linestyle="--", label=f"Simulated: {np.mean(data):.4f}")
        ax.legend(facecolor="#2a2a2a", labelcolor="white")
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel("Density", color="white")
    ax.grid(True, alpha=0.2, color="#444444")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_convergence(results, analytical, title, filename="convergence.png"):
    ns = list(results.keys())
    estimates = [results[n][0] for n in ns]
    errors = [results[n][1] for n in ns]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.patch.set_facecolor("#1e1e1e")
    for ax in [ax1, ax2]:
        style(ax)
    ax1.semilogx(ns, estimates, color="#1f77b4", marker="o", linewidth=1.5, label="MC Estimate")
    ax1.axhline(analytical, color="#ff4d4d", linestyle="--", linewidth=1.5, label=f"Analytical: {analytical:.4f}")
    ax1.set_title(title, color="white", fontsize=12)
    ax1.set_ylabel("Estimate", color="white")
    ax1.legend(facecolor="#2a2a2a", labelcolor="white")
    ax1.grid(True, alpha=0.2, color="#444444")
    ax2.loglog(ns, errors, color="#f0a500", marker="o", linewidth=1.5, label="Absolute Error")
    ax2.set_xlabel("Number of Trials (log scale)", color="white")
    ax2.set_ylabel("Error (log scale)", color="white")
    ax2.set_title("Convergence: Error Decreases as Trials Increase", color="white", fontsize=11)
    ax2.legend(facecolor="#2a2a2a", labelcolor="white")
    ax2.grid(True, alpha=0.2, color="#444444")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

def plot_random_walks(paths, title, filename="random_walks.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#1e1e1e")
    style(ax)
    for path in paths[:100]:
        ax.plot(path, color="#1f77b4", alpha=0.1, linewidth=0.5)
    ax.plot(paths.mean(axis=0), color="#00cc66", linewidth=2, label="Mean path")
    ax.axhline(1.0, color="white", linestyle="--", linewidth=0.8, label="Starting value")
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel("Trading Days", color="white")
    ax.set_ylabel("Normalised Price", color="white")
    ax.legend(facecolor="#2a2a2a", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#444444")
    plt.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path_out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path_out}")