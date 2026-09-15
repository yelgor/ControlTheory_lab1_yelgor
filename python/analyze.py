"""
Practical 1: load UART-logged displacement data and plot y(t).

Expected data format in ../data/data.txt (two columns, space-separated):
    t[s]  y[m]

Example:
    0.0000 0.0000
    0.1000 0.0109
    ...
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.txt"
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Save PuTTY session output to data/data.txt first."
        )

    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(
            f"Expected two columns (t y) in {path}, got shape {data.shape}"
        )

    t = data[:, 0]
    y = data[:, 1]
    return t, y

def calculate_derivative(
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if t.ndim != 1 or y.ndim != 1:
        raise ValueError("t та y мають бути одновимірними масивами")

    if t.size != y.size:
        raise ValueError("t та y повинні мати однакову довжину")

    if t.size < 2:
        raise ValueError("Для похідної потрібно щонайменше дві точки")

    if np.any(np.diff(t) <= 0):
        raise ValueError("Значення t повинні строго зростати")

    dy_dt = np.gradient(y, t, edge_order=1)

    return t.copy(), dy_dt

def plot_displacement(t: np.ndarray, y: np.ndarray, save_path: Path, y_label: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, y, "o-", color="tab:blue", markersize=4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot: {save_path}")
    plt.show()


def main() -> None:
    t, y = load_data(DATA_PATH)
    print(f"Loaded {len(t)} samples from {DATA_PATH}")
    print(np.column_stack((t, y)))
    t_1, v = calculate_derivative(t, y)
    t_2, a = calculate_derivative(t_1, v)

    plot_displacement(t, y, OUTPUT_DIR / "displacement_plot.png",
                      "Displacement (m)", "Залежність переміщення від часу" )
    plot_displacement(t_1, v, OUTPUT_DIR / "velocity_image.png",
                      "Velocity (m/s)", "Залежність швидкості від часу ")
    plot_displacement(t_2, a, OUTPUT_DIR / "axeleration_image.png",
                      "Axeleration (m/s^2)", "Залежінсть прикскорення від часу")
    # TODO (завдання):
    # 1. Обчисліть швидкість і прискорення чисельними похідними.
    # 2. Побудуйте графіки v(t) та a(t), збережіть як .png у цій папці.


if __name__ == "__main__":
    main()
