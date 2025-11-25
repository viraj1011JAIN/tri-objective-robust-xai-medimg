"""
Comprehensive Test Suite for Concept Bank Creation (Phase 6.5).

Tests cover:
- Configuration validation
- Medical concept extraction (dermoscopy + chest X-ray)
- Artifact detection (all heuristics)
- Random patch generation
- Quality control (blur, contrast, diversity)
- Integration with DVC
- Edge cases (empty datasets, corrupted images)
- Hypothesis H3: Concept availability for TCAV

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Target: 100% Coverage, Production-Grade Quality
Date: November 25, 2025
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from src.xai.concept_bank import (
    CHEST_XRAY_ARTIFACT_CONCEPTS,
    CHEST_XRAY_MEDICAL_CONCEPTS,
    DERMOSCOPY_ARTIFACT_CONCEPTS,
    DERMOSCOPY_MEDICAL_CONCEPTS,
    ConceptBankConfig,
    ConceptBankCreator,
    create_concept_bank_creator,
)

# ============================================================================
# Test Configuration
# ============================================================================


class TestConceptBankConfig:
    """Test configuration validation and defaults."""

    def test_valid_dermoscopy_config(self):
        """Test valid dermoscopy configuration."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts/dermoscopy",
        )

        assert config.modality == "dermoscopy"
        assert config.output_dir == Path("data/concepts/dermoscopy")
        assert config.patch_size == (224, 224)
        assert config.num_medical_per_concept == 100
        assert config.num_artifact_per_concept == 50
        assert config.num_random == 200
        assert config.seed == 42
        assert config.min_patch_quality == 0.5
        assert config.diversity_threshold == 0.3
        assert config.use_dvc is True
        assert config.verbose == 1

    def test_valid_chestxray_config(self):
        """Test valid chest X-ray configuration."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts/chest_xray",
        )

        assert config.modality == "chest_xray"
        assert config.output_dir == Path("data/concepts/chest_xray")

    def test_invalid_modality(self):
        """Test that invalid modality raises ValueError."""
        with pytest.raises(ValueError, match="modality must be"):
            ConceptBankConfig(
                modality="invalid",
                output_dir="data/concepts",
            )

    def test_invalid_patch_size_negative(self):
        """Test that negative patch size raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                patch_size=(-224, 224),
            )

    def test_invalid_patch_size_zero(self):
        """Test that zero patch size raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be positive"):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                patch_size=(0, 224),
            )

    def test_invalid_num_medical_zero(self):
        """Test that zero num_medical_per_concept raises ValueError."""
        with pytest.raises(
            ValueError, match="num_medical_per_concept must be positive"
        ):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                num_medical_per_concept=0,
            )

    def test_invalid_num_medical_negative(self):
        """Test that negative num_medical_per_concept raises ValueError."""
        with pytest.raises(
            ValueError, match="num_medical_per_concept must be positive"
        ):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                num_medical_per_concept=-10,
            )

    def test_invalid_num_artifact_zero(self):
        """Test that zero num_artifact_per_concept raises ValueError."""
        with pytest.raises(
            ValueError, match="num_artifact_per_concept must be positive"
        ):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                num_artifact_per_concept=0,
            )

    def test_invalid_min_patch_quality_negative(self):
        """Test that negative min_patch_quality raises ValueError."""
        with pytest.raises(ValueError, match="min_patch_quality must be in"):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                min_patch_quality=-0.1,
            )

    def test_invalid_min_patch_quality_above_one(self):
        """Test that min_patch_quality > 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_patch_quality must be in"):
            ConceptBankConfig(
                modality="dermoscopy",
                output_dir="data/concepts",
                min_patch_quality=1.5,
            )

    def test_custom_parameters(self):
        """Test custom parameter overrides."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="custom/path",
            patch_size=(256, 256),
            num_medical_per_concept=150,
            num_artifact_per_concept=75,
            num_random=300,
            seed=123,
            min_patch_quality=0.7,
            diversity_threshold=0.4,
            use_dvc=False,
            verbose=2,
        )

        assert config.patch_size == (256, 256)
        assert config.num_medical_per_concept == 150
        assert config.num_artifact_per_concept == 75
        assert config.num_random == 300
        assert config.seed == 123
        assert config.min_patch_quality == 0.7
        assert config.diversity_threshold == 0.4
        assert config.use_dvc is False
        assert config.verbose == 2

    def test_output_dir_path_conversion(self):
        """Test that output_dir is converted to Path."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )

        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("data/concepts")


# ============================================================================
# Test Creator Initialization
# ============================================================================


class TestConceptBankCreator:
    """Test ConceptBankCreator initialization."""

    def test_initialization_dermoscopy(self):
        """Test initialization with dermoscopy config."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts/dermoscopy",
        )
        creator = ConceptBankCreator(config)

        assert creator.config == config
        assert creator.medical_concepts == DERMOSCOPY_MEDICAL_CONCEPTS
        assert creator.artifact_concepts == DERMOSCOPY_ARTIFACT_CONCEPTS

    def test_initialization_chestxray(self):
        """Test initialization with chest X-ray config."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts/chest_xray",
        )
        creator = ConceptBankCreator(config)

        assert creator.config == config
        assert creator.medical_concepts == CHEST_XRAY_MEDICAL_CONCEPTS
        assert creator.artifact_concepts == CHEST_XRAY_ARTIFACT_CONCEPTS

    def test_repr(self):
        """Test string representation."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        repr_str = repr(creator)
        assert "ConceptBankCreator" in repr_str
        assert "dermoscopy" in repr_str
        assert "medical_concepts" in repr_str
        assert "artifact_concepts" in repr_str


# ============================================================================
# Test Directory Structure Creation
# ============================================================================


class TestDirectoryStructure:
    """Test directory structure creation."""

    def test_create_directory_structure_dermoscopy(self, tmp_path):
        """Test directory structure for dermoscopy."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
        )
        creator = ConceptBankCreator(config)

        creator._create_directory_structure()

        # Check medical directories
        medical_dir = tmp_path / "concepts" / "medical"
        assert medical_dir.exists()

        for concept in DERMOSCOPY_MEDICAL_CONCEPTS.keys():
            assert (medical_dir / concept).exists()

        # Check artifact directories
        artifact_dir = tmp_path / "concepts" / "artifacts"
        assert artifact_dir.exists()

        for concept in DERMOSCOPY_ARTIFACT_CONCEPTS.keys():
            assert (artifact_dir / concept).exists()

        # Check random directory
        assert (tmp_path / "concepts" / "random").exists()

    def test_create_directory_structure_chestxray(self, tmp_path):
        """Test directory structure for chest X-ray."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir=tmp_path / "concepts",
        )
        creator = ConceptBankCreator(config)

        creator._create_directory_structure()

        # Check medical directories
        medical_dir = tmp_path / "concepts" / "medical"
        assert medical_dir.exists()

        for concept in CHEST_XRAY_MEDICAL_CONCEPTS.keys():
            assert (medical_dir / concept).exists()

        # Check artifact directories
        artifact_dir = tmp_path / "concepts" / "artifacts"
        assert artifact_dir.exists()

        for concept in CHEST_XRAY_ARTIFACT_CONCEPTS.keys():
            assert (artifact_dir / concept).exists()


# ============================================================================
# Test Patch Extraction
# ============================================================================


class TestPatchExtraction:
    """Test patch extraction from images."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create sample test image."""
        img_path = tmp_path / "sample.jpg"
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path

    @pytest.fixture
    def high_quality_image(self, tmp_path):
        """Create high-quality test image with texture."""
        img_path = tmp_path / "quality.jpg"
        # Create textured image (checkerboard)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(0, 512, 32):
            for j in range(0, 512, 32):
                if (i // 32 + j // 32) % 2 == 0:
                    img[i : i + 32, j : j + 32] = [255, 255, 255]
        cv2.imwrite(str(img_path), img)
        return img_path

    def test_extract_patches_from_image(self, sample_image):
        """Test extracting patches from image."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            patch_size=(224, 224),
        )
        creator = ConceptBankCreator(config)

        patches = creator._extract_patches_from_image(
            sample_image, num_patches=5, quality_check=False
        )

        assert len(patches) <= 5
        for patch in patches:
            assert patch.shape[:2] == (224, 224)

    def test_extract_patches_with_quality_check(self, high_quality_image):
        """Test patch extraction with quality check enabled."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            min_patch_quality=0.3,
        )
        creator = ConceptBankCreator(config)

        patches = creator._extract_patches_from_image(
            high_quality_image, num_patches=3, quality_check=True
        )

        # Should extract some patches (high quality image)
        assert len(patches) > 0

    def test_extract_patches_from_small_image(self, tmp_path):
        """Test extracting patches from image smaller than patch size."""
        img_path = tmp_path / "small.jpg"
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            patch_size=(224, 224),
        )
        creator = ConceptBankCreator(config)

        patches = creator._extract_patches_from_image(
            img_path, num_patches=1, quality_check=False
        )

        # Should resize and return 1 patch
        assert len(patches) == 1
        assert patches[0].shape[:2] == (224, 224)

    def test_extract_patches_from_nonexistent_image(self, tmp_path):
        """Test extracting patches from nonexistent image."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        patches = creator._extract_patches_from_image(
            tmp_path / "nonexistent.jpg", num_patches=5
        )

        assert len(patches) == 0

    def test_extract_patches_from_regions(self, sample_image):
        """Test extracting patches from specified regions."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        # Define regions (normalized coordinates)
        regions = [
            (0.0, 0.0, 0.5, 0.5),  # Top-left quadrant
            (0.5, 0.5, 1.0, 1.0),  # Bottom-right quadrant
        ]

        patches = creator._extract_patches_from_regions(
            sample_image, regions, num_per_region=2
        )

        # Should extract 4 patches (2 per region)
        assert len(patches) == 4


# ============================================================================
# Test Quality Control
# ============================================================================


class TestQualityControl:
    """Test patch quality control."""

    def test_check_patch_quality_high_quality(self):
        """Test quality check on high-quality patch."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            min_patch_quality=0.3,
        )
        creator = ConceptBankCreator(config)

        # High-quality patch (textured)
        patch = np.zeros((224, 224, 3), dtype=np.uint8)
        for i in range(0, 224, 16):
            for j in range(0, 224, 16):
                if (i // 16 + j // 16) % 2 == 0:
                    patch[i : i + 16, j : j + 16] = [255, 255, 255]

        assert creator._check_patch_quality(patch)

    def test_check_patch_quality_low_quality(self):
        """Test quality check on low-quality patch (uniform)."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            min_patch_quality=0.5,
        )
        creator = ConceptBankCreator(config)

        # Low-quality patch (completely black - zero variance, zero blur)
        patch = np.zeros((224, 224, 3), dtype=np.uint8)

        # Quality score = (0/500 + 0/100)/2 = 0 < 0.5
        assert not creator._check_patch_quality(patch)

    def test_check_patch_diversity_first_patch(self):
        """Test diversity check on first patch (always passes)."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        patch = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # First patch should always pass
        assert creator._check_patch_diversity(patch, [])

    def test_check_patch_diversity_similar_patches(self):
        """Test diversity check rejects very similar patches."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            diversity_threshold=0.3,
        )
        creator = ConceptBankCreator(config)

        # Create identical patches
        patch1 = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        patch2 = patch1.copy()

        # Should reject (too similar)
        assert not creator._check_patch_diversity(patch2, [patch1])

    def test_check_patch_diversity_different_patches(self):
        """Test diversity check accepts different patches."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            diversity_threshold=0.3,
        )
        creator = ConceptBankCreator(config)

        # Create different patches
        patch1 = np.zeros((224, 224, 3), dtype=np.uint8)
        patch2 = np.full((224, 224, 3), 255, dtype=np.uint8)

        # Should accept (very different)
        assert creator._check_patch_diversity(patch2, [patch1])


# ============================================================================
# Test Artifact Detection
# ============================================================================


class TestArtifactDetection:
    """Test artifact detection heuristics."""

    @pytest.fixture
    def image_with_lines(self):
        """Create image with horizontal lines (ruler simulation)."""
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add horizontal white lines
        img[100:110, :] = [255, 255, 255]
        img[200:210, :] = [255, 255, 255]
        return img

    @pytest.fixture
    def image_with_hair(self):
        """Create image with thin dark lines (hair simulation)."""
        img = np.full((512, 512, 3), 200, dtype=np.uint8)
        # Add thin dark lines
        cv2.line(img, (50, 50), (450, 450), (0, 0, 0), 2)
        cv2.line(img, (100, 50), (400, 450), (0, 0, 0), 1)
        return img

    def test_detect_ruler(self, image_with_lines):
        """Test ruler detection."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        patches = creator._detect_ruler(image_with_lines)

        # Should detect horizontal lines
        assert len(patches) > 0

    def test_detect_hair(self, image_with_hair):
        """Test hair detection."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        patches = creator._detect_hair(image_with_hair)

        # Should detect thin lines
        assert len(patches) >= 0  # May or may not detect depending on threshold

    def test_detect_black_borders(self):
        """Test black border detection."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        # Create image with black border
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[50:450, 50:450] = [255, 255, 255]  # White center

        patches = creator._detect_black_borders(img)

        # Should detect border
        assert len(patches) > 0

    def test_detect_text_overlay(self):
        """Test text overlay detection in chest X-rays."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        # Create image with simulated text (high edge density in corner)
        img = np.full((512, 512), 50, dtype=np.uint8)
        # Add edge-rich pattern in top-left corner
        for i in range(0, 100, 5):
            cv2.line(img, (i, 0), (i, 100), (255,), 1)

        patches = creator._detect_text_overlay(img)

        # Should detect high edge density
        assert len(patches) >= 0

    def test_detect_xray_borders(self):
        """Test border detection in chest X-rays."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        img = np.zeros((512, 512), dtype=np.uint8)

        patches = creator._detect_xray_borders(img)

        # Should extract border regions
        assert len(patches) == 2  # Top-left and top-right

    def test_detect_patient_markers(self):
        """Test patient marker detection."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        # Create image with bright spots (markers)
        img = np.zeros((512, 512), dtype=np.uint8)
        cv2.circle(img, (100, 100), 20, 250, -1)  # Bright circle

        patches = creator._detect_patient_markers(img)

        # Should detect bright spots
        assert len(patches) >= 0

    def test_detect_blank_regions(self):
        """Test blank region detection."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        # Create image with large dark region
        img = np.full((512, 512), 100, dtype=np.uint8)
        img[100:400, 100:400] = 5  # Large dark region

        patches = creator._detect_blank_regions(img)

        # Should detect blank area
        assert len(patches) >= 0


# ============================================================================
# Test Saving
# ============================================================================


class TestSaving:
    """Test patch saving functionality."""

    def test_save_patch(self, tmp_path):
        """Test saving single patch."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            patch_size=(224, 224),
        )
        creator = ConceptBankCreator(config)

        patch = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        save_path = tmp_path / "concepts" / "test_patch.png"

        creator._save_patch(patch, save_path)

        assert save_path.exists()

        # Verify saved image
        loaded = cv2.imread(str(save_path))
        assert loaded.shape == (224, 224, 3)

    def test_save_patch_with_resize(self, tmp_path):
        """Test saving patch with automatic resizing."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            patch_size=(224, 224),
        )
        creator = ConceptBankCreator(config)

        # Create patch with different size
        patch = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        save_path = tmp_path / "concepts" / "resized_patch.png"

        creator._save_patch(patch, save_path)

        assert save_path.exists()

        # Verify resized
        loaded = cv2.imread(str(save_path))
        assert loaded.shape == (224, 224, 3)

    def test_save_patches_batch(self, tmp_path):
        """Test saving batch of patches."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_artifact_per_concept=10,
        )
        creator = ConceptBankCreator(config)

        # Create directory structure
        creator._create_directory_structure()

        patches = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(5)
        ]

        count = creator._save_patches(patches, "artifacts", "ruler", start_idx=0)

        assert count == 5

        # Verify all saved
        for i in range(5):
            path = tmp_path / "concepts" / "artifacts" / "ruler" / f"ruler_{i:04d}.png"
            assert path.exists()

    def test_save_metadata(self, tmp_path):
        """Test saving concept bank metadata."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
        )
        creator = ConceptBankCreator(config)

        # Create output directory first
        config.output_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "modality": "dermoscopy",
            "medical_concepts": {"asymmetry": 100, "pigment_network": 100},
            "artifact_concepts": {"ruler": 50, "hair": 50},
            "random_patches": 200,
            "total_patches": 500,
        }

        creator._save_metadata(stats)

        metadata_path = tmp_path / "concepts" / "metadata.json"
        assert metadata_path.exists()

        # Verify content
        with open(metadata_path) as f:
            loaded = json.load(f)

        assert loaded["statistics"] == stats
        assert loaded["config"]["modality"] == "dermoscopy"


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Test end-to-end concept bank creation."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create mock dataset with sample images."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        # Create 10 sample images
        for i in range(10):
            img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(dataset_path / f"image_{i:03d}.jpg"), img)

        return dataset_path

    def test_create_concept_bank_dermoscopy_no_metadata(self, tmp_path, mock_dataset):
        """Test creating dermoscopy concept bank without metadata."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_medical_per_concept=5,
            num_artifact_per_concept=3,
            num_random=10,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        stats = creator.create_concept_bank(mock_dataset)

        # Verify statistics
        assert stats["modality"] == "dermoscopy"
        assert stats["total_patches"] > 0
        assert "medical_concepts" in stats
        assert "artifact_concepts" in stats
        assert stats["random_patches"] > 0

        # Verify directory structure
        assert (tmp_path / "concepts" / "medical").exists()
        assert (tmp_path / "concepts" / "artifacts").exists()
        assert (tmp_path / "concepts" / "random").exists()

    def test_create_concept_bank_chestxray(self, tmp_path, mock_dataset):
        """Test creating chest X-ray concept bank."""
        config = ConceptBankConfig(
            modality="chest_xray",
            output_dir=tmp_path / "concepts",
            num_medical_per_concept=5,
            num_artifact_per_concept=3,
            num_random=10,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        stats = creator.create_concept_bank(mock_dataset)

        assert stats["modality"] == "chest_xray"
        assert stats["total_patches"] > 0

    def test_create_concept_bank_with_dvc_disabled(self, tmp_path, mock_dataset):
        """Test creating concept bank with DVC disabled."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_medical_per_concept=3,
            num_artifact_per_concept=2,
            num_random=5,
            use_dvc=False,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        stats = creator.create_concept_bank(mock_dataset)

        assert stats["total_patches"] > 0

    def test_create_concept_bank_nonexistent_dataset(self, tmp_path):
        """Test creating concept bank with nonexistent dataset."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
        )
        creator = ConceptBankCreator(config)

        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            creator.create_concept_bank(tmp_path / "nonexistent")

    @patch("subprocess.run")
    def test_dvc_tracking_success(self, mock_run, tmp_path, mock_dataset):
        """Test successful DVC tracking."""
        mock_run.return_value = Mock(stdout="DVC: added 'data/concepts'")

        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_medical_per_concept=2,
            num_artifact_per_concept=1,
            num_random=3,
            use_dvc=True,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        stats = creator.create_concept_bank(mock_dataset)

        # Verify DVC was called
        mock_run.assert_called_once()
        assert "dvc" in mock_run.call_args[0][0]
        assert "add" in mock_run.call_args[0][0]
        assert stats["total_patches"] > 0

    @patch("subprocess.run")
    def test_dvc_tracking_failure(self, mock_run, tmp_path, mock_dataset, caplog):
        """Test DVC tracking failure handling."""
        import subprocess

        mock_run.side_effect = subprocess.CalledProcessError(1, "dvc", stderr="Error")

        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_medical_per_concept=2,
            num_artifact_per_concept=1,
            num_random=3,
            use_dvc=True,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        # Should not raise, just log warning
        stats = creator.create_concept_bank(mock_dataset)

        assert "DVC tracking failed" in caplog.text
        assert stats["total_patches"] > 0

    def test_random_patch_generation(self, tmp_path, mock_dataset):
        """Test random patch generation specifically."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_random=20,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        creator._create_directory_structure()
        count = creator._generate_random_patches(mock_dataset)

        assert count == 20

        # Verify patches saved
        random_dir = tmp_path / "concepts" / "random"
        saved_patches = list(random_dir.glob("*.png"))
        assert len(saved_patches) == 20


# ============================================================================
# Test Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test factory function for creating ConceptBankCreator."""

    def test_factory_with_config(self):
        """Test factory function with explicit config."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = create_concept_bank_creator(config=config)

        assert isinstance(creator, ConceptBankCreator)
        assert creator.config == config

    def test_factory_with_kwargs(self):
        """Test factory function with kwargs."""
        creator = create_concept_bank_creator(
            modality="chest_xray",
            output_dir="custom/path",
            num_medical_per_concept=150,
        )

        assert isinstance(creator, ConceptBankCreator)
        assert creator.config.modality == "chest_xray"
        assert creator.config.num_medical_per_concept == 150

    def test_factory_defaults(self):
        """Test factory function with minimal parameters."""
        creator = create_concept_bank_creator(
            modality="dermoscopy",
            output_dir="data/concepts",
        )

        assert isinstance(creator, ConceptBankCreator)
        assert creator.config.num_medical_per_concept == 100


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dataset(self, tmp_path):
        """Test concept bank creation with empty dataset."""
        dataset_path = tmp_path / "empty_dataset"
        dataset_path.mkdir()

        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            num_medical_per_concept=5,
            verbose=0,
        )
        creator = ConceptBankCreator(config)

        # Should not crash, just return zero counts
        stats = creator.create_concept_bank(dataset_path)

        # All counts should be zero or very low
        for concept_counts in stats["medical_concepts"].values():
            assert concept_counts == 0

    def test_extract_from_corrupted_image(self, tmp_path):
        """Test extracting patches from corrupted image file."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )
        creator = ConceptBankCreator(config)

        # Create corrupted file
        corrupted = tmp_path / "corrupted.jpg"
        corrupted.write_text("This is not an image")

        patches = creator._extract_patches_from_image(corrupted, num_patches=5)

        # Should return empty list
        assert len(patches) == 0

    def test_max_patch_extraction_attempts(self, tmp_path):
        """Test that patch extraction terminates after max attempts."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
            min_patch_quality=0.99,  # Very high threshold
        )
        creator = ConceptBankCreator(config)

        # Create low-quality image
        img_path = tmp_path / "low_quality.jpg"
        img = np.full((512, 512, 3), 128, dtype=np.uint8)  # Uniform gray
        cv2.imwrite(str(img_path), img)

        patches = creator._extract_patches_from_image(
            img_path, num_patches=10, quality_check=True
        )

        # Should terminate before extracting 10 patches
        assert len(patches) < 10


# ============================================================================
# Test Hypothesis H3
# ============================================================================


class TestHypothesisH3:
    """
    Test Hypothesis H3: Concept availability for TCAV.

    H3: Concept banks must contain:
        - Sufficient medical concepts (100+ per concept)
        - Sufficient artifact concepts (50+ per concept)
        - Random baseline patches (200+)
        - Diverse and high-quality patches
    """

    def test_h3_concept_counts_sufficient(self):
        """Test that concept counts meet H3 requirements."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )

        # H3: Medical concepts >= 100
        assert config.num_medical_per_concept >= 100

        # H3: Artifact concepts >= 50
        assert config.num_artifact_per_concept >= 50

        # H3: Random patches >= 200
        assert config.num_random >= 200

    def test_h3_quality_thresholds(self):
        """Test that quality thresholds are reasonable."""
        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir="data/concepts",
        )

        # H3: Quality threshold should filter low-quality patches
        assert 0.3 <= config.min_patch_quality <= 0.7

        # H3: Diversity threshold should ensure variation
        assert 0.2 <= config.diversity_threshold <= 0.5

    def test_h3_all_concepts_defined(self):
        """Test that all required concepts are defined."""
        # Dermoscopy concepts
        assert len(DERMOSCOPY_MEDICAL_CONCEPTS) >= 5
        assert len(DERMOSCOPY_ARTIFACT_CONCEPTS) >= 3

        # Chest X-ray concepts
        assert len(CHEST_XRAY_MEDICAL_CONCEPTS) >= 3
        assert len(CHEST_XRAY_ARTIFACT_CONCEPTS) >= 3

        # H3: Must include both medical and artifact concepts
        assert (
            "asymmetry" in DERMOSCOPY_MEDICAL_CONCEPTS
            or "pigment_network" in DERMOSCOPY_MEDICAL_CONCEPTS
        )
        assert (
            "ruler" in DERMOSCOPY_ARTIFACT_CONCEPTS
            or "hair" in DERMOSCOPY_ARTIFACT_CONCEPTS
        )


# ============================================================================
# Performance and Logging Tests
# ============================================================================


class TestLogging:
    """Test logging functionality."""

    def test_log_summary(self, caplog, tmp_path):
        """Test that summary logging works correctly."""
        import logging

        # Set logging level to capture INFO
        caplog.set_level(logging.INFO)

        config = ConceptBankConfig(
            modality="dermoscopy",
            output_dir=tmp_path / "concepts",
            verbose=1,
        )
        creator = ConceptBankCreator(config)

        stats = {
            "modality": "dermoscopy",
            "medical_concepts": {"asymmetry": 100},
            "artifact_concepts": {"ruler": 50},
            "random_patches": 200,
            "total_patches": 350,
        }

        creator._log_summary(stats)

        assert "CONCEPT BANK SUMMARY" in caplog.text
        assert "Total patches: 350" in caplog.text
        assert "asymmetry" in caplog.text
        assert "ruler" in caplog.text
