"""Create dummy ISIC 2018 data for testing."""

import csv
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("F:/data/isic_2018")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Create dummy images
img_dir = DATA_DIR / "ISIC2018_Task3_Training_Input"
img_dir.mkdir(exist_ok=True)

logger.info("Creating 100 dummy images...")
for i in range(100):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(img_dir / f"ISIC_000{i:04d}.jpg")

# Create dummy labels
csv_dir = DATA_DIR / "ISIC2018_Task3_Training_GroundTruth"
csv_dir.mkdir(exist_ok=True)

classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

with open(csv_dir / "ISIC2018_Task3_Training_GroundTruth.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image"] + classes)

    for i in range(100):
        label = [0] * 7
        label[i % 7] = 1
        writer.writerow([f"ISIC_000{i:04d}"] + label)

print("✓ Dummy dataset created for testing")
