"""
CRITICAL DATA VERIFICATION SCRIPT
==================================

This script verifies that REAL medical imaging data exists and can be loaded.
Run this BEFORE retraining to confirm the fix will work.

Author: Viraj Pankaj Jain
Date: November 23, 2025
"""

import sys
from pathlib import Path

# Add project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.isic import ISICDataset
from src.datasets.transforms import get_test_transforms, get_train_transforms


def verify_data_exists():
    """Step 1: Verify data files exist on disk."""
    print("=" * 80)
    print("STEP 1: Verifying Data Files Exist")
    print("=" * 80)

    data_root = ROOT / "data" / "processed" / "isic2018"
    csv_path = data_root / "metadata_processed.csv"

    checks = {
        "Data root directory": data_root.exists(),
        "Metadata CSV file": csv_path.exists(),
    }

    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            print(f"   Expected: {data_root if 'root' in check_name else csv_path}")

    if not all(checks.values()):
        print("\n❌ DATA FILES MISSING - Run preprocessing first")
        return False

    print("\n✅ All data files found")
    return True


def verify_dataset_loading():
    """Step 2: Verify datasets can be loaded."""
    print("\n" + "=" * 80)
    print("STEP 2: Verifying Dataset Loading")
    print("=" * 80)

    data_root = ROOT / "data" / "processed" / "isic2018"
    csv_path = data_root / "metadata_processed.csv"

    try:
        # Load train dataset
        train_transforms = get_train_transforms(dataset="isic", image_size=224)
        train_data = ISICDataset(
            root=str(data_root),
            split="train",
            csv_path=str(csv_path),
            transforms=train_transforms,
        )

        # Load val dataset
        test_transforms = get_test_transforms(dataset="isic", image_size=224)
        val_data = ISICDataset(
            root=str(data_root),
            split="val",
            csv_path=str(csv_path),
            transforms=test_transforms,
        )

        # Load test dataset
        test_data = ISICDataset(
            root=str(data_root),
            split="test",
            csv_path=str(csv_path),
            transforms=test_transforms,
        )

        print(f"✅ Train dataset: {len(train_data)} samples")
        print(f"✅ Val dataset: {len(val_data)} samples")
        print(f"✅ Test dataset: {len(test_data)} samples")
        print(f"✅ Num classes: {train_data.num_classes}")

        return train_data, val_data, test_data

    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return None, None, None


def verify_sample_loading(train_data):
    """Step 3: Verify individual samples can be loaded."""
    print("\n" + "=" * 80)
    print("STEP 3: Verifying Sample Loading")
    print("=" * 80)

    if train_data is None:
        print("❌ Cannot verify samples - dataset not loaded")
        return False

    try:
        import torch

        # Load first sample
        img, label, meta = train_data[0]

        print(f"✅ Image shape: {img.shape}")
        print(f"✅ Image dtype: {img.dtype}")
        print(f"✅ Image range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"✅ Label: {label} (type: {type(label)})")
        print(f"✅ Metadata keys: {list(meta.keys())}")
        print(f"✅ Image path: {meta.get('path', 'N/A')}")

        # Verify image is correct shape for model
        assert img.shape == (3, 224, 224), f"Wrong shape: {img.shape}"
        assert 0 <= label < 7, f"Invalid label: {label}"
        assert img.dtype == torch.float32, f"Wrong dtype: {img.dtype}"

        # Check if this is REAL data or synthetic
        is_synthetic = (img == torch.randn_like(img)).all()
        if is_synthetic:
            print("\n⚠️  WARNING: Image appears to be synthetic random noise")
            return False

        print("\n✅ Sample loading verified - Images are REAL data")
        return True

    except Exception as e:
        print(f"❌ Sample loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_class_distribution(train_data):
    """Step 4: Verify class distribution is reasonable."""
    print("\n" + "=" * 80)
    print("STEP 4: Verifying Class Distribution")
    print("=" * 80)

    if train_data is None:
        print("❌ Cannot verify distribution - dataset not loaded")
        return False

    try:
        from collections import Counter

        import torch

        # Sample labels from first 1000 samples (faster)
        labels = []
        for i in range(min(1000, len(train_data))):
            _, label, _ = train_data[i]
            labels.append(int(label) if isinstance(label, torch.Tensor) else label)

        counts = Counter(labels)
        total = len(labels)

        print("Class distribution (first 1000 samples):")
        for class_id in sorted(counts.keys()):
            count = counts[class_id]
            pct = 100.0 * count / total
            print(f"  Class {class_id}: {count:4d} samples ({pct:5.1f}%)")

        # Check if distribution is suspiciously uniform (sign of random labels)
        min_pct = min(100.0 * v / total for v in counts.values())
        max_pct = max(100.0 * v / total for v in counts.values())

        if max_pct - min_pct < 5.0:
            print(
                "\n⚠️  WARNING: Distribution is suspiciously uniform (possible random labels)"
            )
            return False

        print("\n✅ Class distribution verified - Non-uniform (REAL labels)")
        return True

    except Exception as e:
        print(f"❌ Distribution check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    import torch

    print("\n" + "🔍 " * 40)
    print("CRITICAL DATA VERIFICATION")
    print("Verifying REAL medical imaging data is loaded correctly")
    print("🔍 " * 40 + "\n")

    # Run all checks
    results = {}

    results["Files exist"] = verify_data_exists()
    if not results["Files exist"]:
        print("\n" + "❌" * 40)
        print("VERIFICATION FAILED: Data files not found")
        print("Run preprocessing before training")
        print("❌" * 40)
        return False

    train_data, val_data, test_data = verify_dataset_loading()
    results["Dataset loading"] = train_data is not None
    if not results["Dataset loading"]:
        print("\n" + "❌" * 40)
        print("VERIFICATION FAILED: Cannot load datasets")
        print("❌" * 40)
        return False

    results["Sample loading"] = verify_sample_loading(train_data)
    results["Class distribution"] = verify_class_distribution(train_data)

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = all(results.values())

    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")

    if all_passed:
        print("\n" + "🎯" * 40)
        print("✅ ALL CHECKS PASSED - READY TO TRAIN ON REAL DATA")
        print("🎯" * 40)
        print("\nNext steps:")
        print("1. Fix create_dataloaders() in src/training/train_baseline.py")
        print("2. Run: python scripts/training/train_baseline.py --seed 42")
        print("3. Verify training accuracy improves (should reach ~85%)")
    else:
        print("\n" + "❌" * 40)
        print("VERIFICATION FAILED - DO NOT PROCEED WITH TRAINING")
        print("Fix the issues above first")
        print("❌" * 40)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
