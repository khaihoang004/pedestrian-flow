import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUTPUT_PATH = Path("models/empirical.py")


def load_image(image_path: str):
    try:
        return mpimg.imread(image_path)
    except FileNotFoundError:
        print(f"Error: image file not found: {image_path}")
        sys.exit(1)


def get_calibration_points(ax):
    print("=" * 60)
    print("Step 1: Axis calibration")
    print("Click the following 4 reference points in order:")
    print("  1. X-axis minimum point")
    print("  2. X-axis maximum point")
    print("  3. Y-axis minimum point")
    print("  4. Y-axis maximum point")

    ax.set_title("Step 1: Click X_min, X_max, Y_min, Y_max")
    points = plt.ginput(4, timeout=-1, show_clicks=True)

    if len(points) < 4:
        print("Error: fewer than 4 calibration points were selected.")
        sys.exit(1)

    return points


def get_axis_values():
    print("\nEnter the real values of the selected reference points:")
    x_min = float(input("  X_min: "))
    x_max = float(input("  X_max: "))
    y_min = float(input("  Y_min: "))
    y_max = float(input("  Y_max: "))
    return x_min, x_max, y_min, y_max


def build_pixel_to_real_transform(ref_points, axis_values):
    px_x_min, _ = ref_points[0]
    px_x_max, _ = ref_points[1]
    _, px_y_min = ref_points[2]
    _, px_y_max = ref_points[3]

    x_min, x_max, y_min, y_max = axis_values

    def px_to_real(px, py):
        real_x = x_min + (px - px_x_min) * (x_max - x_min) / (px_x_max - px_x_min)
        real_y = y_min + (py - px_y_min) * (y_max - y_min) / (px_y_max - px_y_min)
        return real_x, real_y

    return px_to_real


def get_data_points(ax):
    print("\n" + "=" * 60)
    print("Step 2: Data extraction")
    print("Left-click to select data points on the plot.")
    print("Press Enter or middle-click to finish.")

    ax.set_title("Step 2: Click data points, then press Enter to finish")
    points = plt.ginput(-1, timeout=-1, show_clicks=True, mouse_stop=2)

    if not points:
        print("No data points were selected.")
        sys.exit(0)

    return points


def convert_points(data_points, px_to_real):
    rho_values = []
    velocity_values = []

    for px, py in data_points:
        rho, velocity = px_to_real(px, py)
        rho_values.append(rho)
        velocity_values.append(velocity)

    return np.array(rho_values), np.array(velocity_values)


def save_numpy_arrays(rho_values, velocity_values):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rho_str = ", ".join(f"{value:.2f}" for value in rho_values)
    velocity_str = ", ".join(f"{value:.2f}" for value in velocity_values)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("import numpy as np\n\n")
        f.write("EMPIRICAL_RHO = np.array([\n")
        f.write(f"    {rho_str}\n")
        f.write("])\n\n")
        f.write("EMPIRICAL_V = np.array([\n")
        f.write(f"    {velocity_str}\n")
        f.write("])\n")

    print("\n" + "=" * 60)
    print(f"Saved extracted data to: {OUTPUT_PATH}")
    print("=" * 60)


def main():
    if len(sys.argv) > 2:
        print("Usage: python -m models.plot_digitizer [image_path]")
        sys.exit(1)

    image_path = sys.argv[1] if len(sys.argv) == 2 else "models/empirical_data.png"
    image = load_image(image_path)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.axis("off")

    ref_points = get_calibration_points(ax)
    axis_values = get_axis_values()
    px_to_real = build_pixel_to_real_transform(ref_points, axis_values)

    data_points = get_data_points(ax)
    plt.close(fig)

    rho_values, velocity_values = convert_points(data_points, px_to_real)
    save_numpy_arrays(rho_values, velocity_values)

if __name__ == "__main__":
    main()