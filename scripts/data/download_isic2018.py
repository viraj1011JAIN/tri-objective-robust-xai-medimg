"""
Download ISIC 2018 dataset.

NOTE: ISIC datasets require registration at https://challenge.isic-archive.com/
This script provides download links and instructions.
"""

import logging
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "isic2018"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def detect_dataset_structure(source_path: Path) -> dict:
    """Auto-detect dataset structure in source directory."""
    structure = {"images_dir": None, "csv_file": None, "test_dir": None}

    # Look for image directories
    possible_img_dirs = [
        "ISIC2018_Task3_Training_Input",
        "Training_Input",
        "images",
        "train",
    ]

    for dirname in possible_img_dirs:
        img_path = source_path / dirname
        if img_path.exists() and img_path.is_dir():
            structure["images_dir"] = img_path
            logger.info(f"Found images directory: {img_path}")
            break

    # If not found, check if images are directly in source_path
    if structure["images_dir"] is None:
        jpg_files = list(source_path.glob("*.jpg"))
        if jpg_files:
            structure["images_dir"] = source_path
            logger.info(f"Images found directly in: {source_path}")

    # Look for CSV file
    possible_csv_names = [
        "ISIC2018_Task3_Training_GroundTruth.csv",
        "GroundTruth.csv",
        "labels.csv",
        "train.csv",
    ]

    for csvname in possible_csv_names:
        csv_path = source_path / csvname
        if csv_path.exists():
            structure["csv_file"] = csv_path
            logger.info(f"Found CSV file: {csv_path}")
            break

    # Also check in subdirectories
    if structure["csv_file"] is None:
        for csv_path in source_path.rglob("*.csv"):
            if "ground" in csv_path.name.lower() or "label" in csv_path.name.lower():
                structure["csv_file"] = csv_path
                logger.info(f"Found CSV file: {csv_path}")
                break

    # Look for test directory (optional)
    possible_test_dirs = ["ISIC2018_Task3_Test_Input", "Test_Input", "test"]

    for dirname in possible_test_dirs:
        test_path = source_path / dirname
        if test_path.exists() and test_path.is_dir():
            structure["test_dir"] = test_path
            logger.info(f"Found test directory: {test_path}")
            break

    return structure


def copy_from_external_source(source_path: Path) -> bool:
    """Copy dataset from external location with auto-detection."""
    if not source_path.exists():
        logger.error(f"Source path does not exist: {source_path}")
        return False

    logger.info(f"Detecting dataset structure in: {source_path}")
    structure = detect_dataset_structure(source_path)

    if not structure["images_dir"] or not structure["csv_file"]:
        logger.error("Could not detect dataset structure!")
        logger.error(f"Images dir: {structure['images_dir']}")
        logger.error(f"CSV file: {structure['csv_file']}")

        # List what's in the source directory
        logger.info(f"Contents of {source_path}:")
        for item in source_path.iterdir():
            logger.info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")

        return False

    logger.info(f"Copying dataset to: {DATA_DIR}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy training images
    dst_images = DATA_DIR / "ISIC2018_Task3_Training_Input"

    if dst_images.exists():
        logger.info("Training images already exist, skipping...")
    else:
        logger.info(f"Copying training images from: {structure['images_dir']}")
        shutil.copytree(structure["images_dir"], dst_images)
        num_copied = len(list(dst_images.glob("*.jpg")))
        logger.info(f"✓ Copied {num_copied} training images")

    # Copy ground truth CSV
    dst_gt_dir = DATA_DIR / "ISIC2018_Task3_Training_GroundTruth"
    dst_gt_dir.mkdir(parents=True, exist_ok=True)
    dst_csv = dst_gt_dir / "ISIC2018_Task3_Training_GroundTruth.csv"

    if dst_csv.exists():
        logger.info("Ground truth CSV already exists, skipping...")
    else:
        logger.info(f"Copying ground truth from: {structure['csv_file']}")
        shutil.copy2(structure["csv_file"], dst_csv)
        logger.info("✓ Copied ground truth CSV")

    # Copy test images if available
    if structure["test_dir"]:
        dst_test = DATA_DIR / "ISIC2018_Task3_Test_Input"
        if dst_test.exists():
            logger.info("Test images already exist, skipping...")
        else:
            logger.info(f"Copying test images from: {structure['test_dir']}")
            shutil.copytree(structure["test_dir"], dst_test)
            num_test = len(list(dst_test.glob("*.jpg")))
            logger.info(f"✓ Copied {num_test} test images")

    return True


def create_symlink(source_path: Path) -> bool:
    """Create symbolic link to external dataset (faster than copying)."""
    if not source_path.exists():
        logger.error(f"Source path does not exist: {source_path}")
        return False

    logger.info(f"Creating symbolic link from: {source_path}")
    logger.info(f"To: {DATA_DIR}")

    try:
        if DATA_DIR.exists():
            if DATA_DIR.is_symlink():
                logger.info("Symbolic link already exists")
                return True
            else:
                logger.warning("Data directory exists but is not a symlink")
                return False

        DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        DATA_DIR.symlink_to(source_path, target_is_directory=True)
        logger.info("✓ Symbolic link created successfully")
        return True

    except OSError as e:
        logger.error(f"Failed to create symlink: {e}")
        logger.info("Falling back to copying files...")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file."""
    try:
        logger.info(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return False


def extract_existing_zips():
    """Extract ZIP files if they exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = [
        "ISIC2018_Task3_Training_Input.zip",
        "ISIC2018_Task3_Training_GroundTruth.zip",
        "ISIC2018_Task3_Test_Input.zip",
    ]

    found_zips = []
    for zip_name in zip_files:
        zip_path = DATA_DIR / zip_name
        if zip_path.exists():
            found_zips.append(zip_path)

    if not found_zips:
        return False

    logger.info(f"Found {len(found_zips)} ZIP file(s) in {DATA_DIR}")

    for zip_path in found_zips:
        if extract_zip(zip_path, DATA_DIR):
            logger.info(f"✓ Extracted: {zip_path.name}")
        else:
            logger.error(f"✗ Failed to extract: {zip_path.name}")

    return True


def verify_dataset():
    """Verify dataset files exist."""
    required_paths = [
        DATA_DIR / "ISIC2018_Task3_Training_Input",
        DATA_DIR
        / "ISIC2018_Task3_Training_GroundTruth"
        / "ISIC2018_Task3_Training_GroundTruth.csv",
    ]

    # Alternative CSV location
    alt_csv = DATA_DIR / "ISIC2018_Task3_Training_GroundTruth.csv"
    if not required_paths[1].exists() and alt_csv.exists():
        required_paths[1] = alt_csv

    all_exist = all(p.exists() for p in required_paths)

    if all_exist:
        logger.info("=" * 80)
        logger.info("✓ DATASET SUCCESSFULLY VERIFIED")
        logger.info("=" * 80)

        # Count images
        img_dir = DATA_DIR / "ISIC2018_Task3_Training_Input"
        num_images = len(list(img_dir.glob("*.jpg")))
        logger.info(f"Found {num_images} training images")

        return True
    else:
        logger.warning("Dataset verification incomplete")
        logger.warning("Missing files:")
        for path in required_paths:
            if not path.exists():
                logger.warning(f"  - {path}")
        return False


def main():
    """Main download/extraction logic."""
    logger.info("=" * 80)
    logger.info("ISIC 2018 DATASET SETUP")
    logger.info("=" * 80)

    # Check if dataset already exists
    if verify_dataset():
        logger.info("Dataset already available and verified!")
        return True

    # Try to link/copy from external source
    external_source = Path("F:/data/isic_2018")

    if not external_source.exists():
        logger.error(f"External dataset path not found: {external_source}")
        logger.error("Please update the path in the script or move the dataset")
        return False

    logger.info(f"Found dataset at: {external_source}")
    logger.info("")
    logger.info("Setup method:")
    logger.info("  Copying files (reliable, uses disk space)")
    logger.info("")

    # Use copy method (more reliable on Windows)
    if copy_from_external_source(external_source):
        if verify_dataset():
            logger.info("=" * 80)
            logger.info("✓ DATASET SETUP COMPLETE")
            logger.info("=" * 80)
            return True

    # Try to extract existing ZIP files
    logger.info("Checking for ZIP files...")
    if extract_existing_zips():
        if verify_dataset():
            return True

    logger.error("Dataset setup failed!")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
