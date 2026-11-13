
from pathlib import Path
from PIL import Image
import numpy as np

FILES = {
    "Original": "original.jpg",
    "Transparent body mask": "transparent-mask.png",
    "Overlay": "overlay.jpg",
}

def inspect_image(label, filename):
    path = Path(filename)

    print("\n" + "=" * 55)
    print(label.upper())
    print("=" * 55)

    if not path.exists():
        print(f"ERROR: {filename} was not found.")
        return None

    try:
        image = Image.open(path)
        image.load()

        print(f"File: {path.resolve()}")
        print(f"Format: {image.format}")
        print(f"Dimensions: {image.width} x {image.height}")
        print(f"Mode: {image.mode}")

        result = {
            "width": image.width,
            "height": image.height,
        }

        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            alpha = np.array(rgba)[:, :, 3]

            transparent = np.count_nonzero(alpha == 0)
            opaque = np.count_nonzero(alpha == 255)
            partial = np.count_nonzero(
                (alpha > 0) & (alpha < 255)
            )

            total = alpha.size

            print("\nTransparency analysis:")
            print(f"Fully transparent: {transparent / total:.2%}")
            print(f"Fully opaque: {opaque / total:.2%}")
            print(f"Partially transparent: {partial / total:.2%}")

            visible = alpha > 0

            if np.any(visible):
                ys, xs = np.where(visible)

                print(
                    "Visible-content bounding box: "
                    f"x={xs.min()}, y={ys.min()}, "
                    f"width={xs.max() - xs.min() + 1}, "
                    f"height={ys.max() - ys.min() + 1}"
                )

            result["alpha_coverage"] = np.count_nonzero(visible) / total

        if label == "Original":
            rgb = np.array(image.convert("RGB"))

            print("\nImage pixel analysis:")
            print(f"Average brightness: {rgb.mean():.2f}")
            print(f"Image file size: {path.stat().st_size / 1024:.2f} KB")

        if label == "Transparent body mask":
            rgba = np.array(image.convert("RGBA"))
            alpha = rgba[:, :, 3]
            foreground = alpha > 0

            print("\nBody-mask analysis:")
            print(f"Foreground coverage: {foreground.mean():.2%}")

            if np.any(foreground):
                ys, xs = np.where(foreground)

                print(
                    "Foreground bounding box: "
                    f"x={xs.min()}, y={ys.min()}, "
                    f"width={xs.max() - xs.min() + 1}, "
                    f"height={ys.max() - ys.min() + 1}"
                )

            # Count connected components in the alpha mask.
            try:
                import cv2

                binary = foreground.astype(np.uint8)

                count, labels, stats, centroids = (
                    cv2.connectedComponentsWithStats(
                        binary, connectivity=8
                    )
                )

                components = count - 1

                print(f"Connected foreground components: {components}")

                if components:
                    areas = sorted(
                        stats[1:, cv2.CC_STAT_AREA],
                        reverse=True
                    )

                    print(f"Largest component area: {areas[0]} pixels")

                    if len(areas) > 1:
                        print(
                            "Second-largest component area: "
                            f"{areas[1]} pixels"
                        )

            except ImportError:
                print("OpenCV unavailable; component analysis skipped.")

        if label == "Overlay":
            rgb = np.array(image.convert("RGB"))

            print("\nOverlay pixel analysis:")
            print(f"Average RGB: {rgb.mean(axis=(0, 1)).tolist()}")

        return result

    except Exception as error:
        print(f"ERROR inspecting {filename}: {error}")
        return None


results = {}

for label, filename in FILES.items():
    results[label] = inspect_image(label, filename)

print("\n" + "=" * 55)
print("DIMENSION COMPARISON")
print("=" * 55)

valid = {
    name: result
    for name, result in results.items()
    if result is not None
}

if len(valid) == 3:
    dimensions = {
        (result["width"], result["height"])
        for result in valid.values()
    }

    if len(dimensions) == 1:
        print("PASS: All three images have matching dimensions.")
    else:
        print("WARNING: Image dimensions do not match.")

        for name, result in valid.items():
            print(
                f"{name}: "
                f"{result['width']} x {result['height']}"
            )
else:
    print("WARNING: One or more preview files are missing.")

print("\nDiagnostic inspection completed.")
