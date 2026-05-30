# download_coco_subset.py

import os
import json
import urllib.request
import zipfile
from shutil import copyfile
from PIL import Image

os.makedirs("data/raw/train2017_subset", exist_ok=True)

# 1. Download the captions file (full)
caption_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
urllib.request.urlretrieve(caption_url, "data/raw/annotations.zip")

# Extract only the captions JSON
with zipfile.ZipFile("data/raw/annotations.zip", 'r') as zip_ref:
    zip_ref.extract("annotations/captions_train2017.json", "data/raw/")

# 2. Download image IDs used in captions
with open("data/raw/annotations/captions_train2017.json", 'r') as f:
    data = json.load(f)

image_ids = list({ann["image_id"] for ann in data["annotations"]})
subset_ids = image_ids[:1000]  # First 1000 images

# 3. Download each image separately (only ~300MB total)
print("⬇️ Downloading 1000 images...")
for i, img_id in enumerate(subset_ids):
    url = f"http://images.cocodataset.org/train2017/{img_id:012d}.jpg"
    dest = f"data/raw/train2017_subset/{img_id:012d}.jpg"
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
    if (i + 1) % 100 == 0:
        print(f"{i+1}/1000 images done")

# 4. Save subset caption file
subset_annots = {
    "annotations": [ann for ann in data["annotations"] if ann["image_id"] in subset_ids]
}
with open("data/raw/captions_subset.json", "w") as f:
    json.dump(subset_annots, f)

print("✅ Done. Images and captions saved in data/raw/")
