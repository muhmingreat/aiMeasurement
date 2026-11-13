import json
import base64
from pathlib import Path

json_path = input("Enter the full path to your downloaded JSON file: ").strip()
json_path = Path(json_path.strip('"'))

if not json_path.exists():
    print("Error: JSON file not found.")
    raise SystemExit(1)

with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

previews = data.get("previews", {})

images = {
    "original": "original.jpg",
    "transparent_body_mask": "transparent-mask.png",
    "overlay": "overlay.jpg",
}

for key, filename in images.items():
    preview = previews.get(key)

    if not preview or not preview.get("data_uri"):
        print(f"Missing preview: {key}")
        continue

    data_uri = preview["data_uri"]

    if "," not in data_uri:
        print(f"Invalid image data: {key}")
        continue
    

    image_data = data_uri.split(",", 1)[1]
    image_bytes = base64.b64decode(image_data)

    output_path = Path(filename)
    output_path.write_bytes(image_bytes)

    print(f"Saved: {output_path.resolve()}")

print("\nImage extraction completed.")