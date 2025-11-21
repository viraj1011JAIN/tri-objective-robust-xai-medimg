"""
Tests for ViTB16Classifier (src/models/vit.py).

Covers:
- Initialization (RGB + grayscale, basic arg validation)
- Forward pass and logits shape
- return_features=True (incl. attention_rollout)
- get_feature_maps with subset names
- get_embedding
- get_attention_rollout
- get_layer_output_shapes sanity checks
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from src.models.vit import ViTB16Classifier

torch.manual_seed(0)


def _make_model(**kwargs) -> ViTB16Classifier:
    """
    Helper to construct ViTB16Classifier or skip tests if ViT is unavailable.

    We always use pretrained=False in tests to avoid network downloads.
    """
    try:
        return ViTB16Classifier(pretrained=False, **kwargs)
    except RuntimeError as exc:
        pytest.skip(f"ViT-B/16 not available in this torchvision build: {exc}")


class TestViTB16Basics:
    """Basic construction and argument validation."""

    def test_initialization_default(self) -> None:
        """Default RGB config."""
        model = _make_model(num_classes=7)

        assert model.num_classes == 7
        assert model.in_channels == 3
        assert model.global_pool_type in {"cls", "mean"}
        assert model.embedding_dim > 0
        assert isinstance(model.get_classifier(), torch.nn.Module)

    def test_initialization_grayscale_and_mean_pool(self) -> None:
        """Grayscale input, dropout enabled, mean patch pooling."""
        model = _make_model(
            num_classes=5,
            in_channels=1,
            dropout=0.3,
            global_pool="mean",
        )

        assert model.num_classes == 5
        assert model.in_channels == 1
        assert pytest.approx(model.dropout_prob) == 0.3
        assert model.global_pool_type == "mean"

    def test_initialization_invalid_args(self) -> None:
        """Bad arguments should raise ValueError."""
        # NOTE: BaseModel error message is:
        # "num_classes must be a positive integer, got 0"
        with pytest.raises(ValueError, match="num_classes must be"):
            _make_model(num_classes=0)

        with pytest.raises(ValueError, match="in_channels must be positive"):
            _make_model(num_classes=2, in_channels=0)

        with pytest.raises(ValueError, match="dropout must be in"):
            _make_model(num_classes=2, dropout=-0.1)

        with pytest.raises(ValueError, match="Unsupported global_pool"):
            _make_model(num_classes=2, global_pool="invalid")

        with pytest.raises(ValueError, match="image_size must be positive"):
            _make_model(num_classes=2, image_size=0)


class TestViTB16Forward:
    """Forward pass behaviour."""

    def test_forward_logits_shape(self) -> None:
        """Forward pass returns correct logits shape."""
        model = _make_model(num_classes=7)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert isinstance(logits, Tensor)
        assert logits.shape == (2, 7)

    def test_forward_grayscale(self) -> None:
        """Grayscale inputs (1 channel) should work via conv_proj adaptation."""
        model = _make_model(num_classes=3, in_channels=1)
        model.eval()

        x = torch.randn(4, 1, 224, 224)
        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (4, 3)

    def test_forward_with_features_flag(self) -> None:
        """return_features=True should include attention_rollout and maps."""
        model = _make_model(num_classes=4)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            logits, feats = model(x, return_features=True)

        # Logits shape
        assert logits.shape == (1, 4)

        # Basic keys
        assert "cls_token" in feats
        assert "patch_tokens" in feats
        assert "tokens" in feats

        cls_token = feats["cls_token"]
        patch_tokens = feats["patch_tokens"]
        tokens = feats["tokens"]

        # Token shapes
        assert cls_token.ndim == 2  # (B, D)
        assert patch_tokens.ndim == 3  # (B, N, D)
        assert tokens.ndim == 3  # (B, 1+N, D)
        assert tokens.shape[1] == patch_tokens.shape[1] + 1
        assert cls_token.shape == (tokens.shape[0], tokens.shape[2])

        # If patch grid is square, we should have patch_map + attention_rollout
        if "patch_map" in feats:
            patch_map = feats["patch_map"]
            attn_rollout = feats["attention_rollout"]

            assert patch_map.ndim == 4  # (B, D, H, W)
            assert attn_rollout.ndim == 4  # (B, 1, H, W)

            b, n_patches, _ = patch_tokens.shape
            grid = int(math.sqrt(n_patches))
            assert grid * grid == n_patches
            assert patch_map.shape[0] == b
            assert patch_map.shape[-2:] == (grid, grid)
            assert attn_rollout.shape == (b, 1, grid, grid)

    def test_forward_invalid_input_dim_raises(self) -> None:
        """Non-4D input should raise a ValueError."""
        model = _make_model(num_classes=2)

        bad_x = torch.randn(3, 224, 224)  # Missing batch or channel dim
        with pytest.raises(ValueError, match="Expected 4D input"):
            _ = model(bad_x)


class TestViTB16FeatureMaps:
    """Feature map extraction helpers."""

    def test_get_feature_maps_default_tokens_only(self) -> None:
        """Default get_feature_maps should return tokens only."""
        model = _make_model(num_classes=2)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feats = model.get_feature_maps(x)

        assert list(feats.keys()) == ["tokens"]
        tokens = feats["tokens"]
        assert tokens.ndim == 3  # (B, 1+N, D)
        assert tokens.shape[0] == 1

    def test_get_feature_maps_subset_layers(self) -> None:
        """Request a subset of feature layers (incl. attention_rollout)."""
        model = _make_model(num_classes=2)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            feats = model.get_feature_maps(
                x,
                layer_names=["cls_token", "attention_rollout"],
            )

        assert "cls_token" in feats
        assert "attention_rollout" in feats

        cls_token = feats["cls_token"]
        attn_rollout = feats["attention_rollout"]

        assert cls_token.ndim == 2
        assert cls_token.shape[0] == 1

        assert attn_rollout.ndim == 4
        assert attn_rollout.shape[0] == 1
        assert attn_rollout.shape[1] == 1

    def test_get_feature_maps_invalid_layer_raises(self) -> None:
        """Requesting an unknown layer should raise ValueError."""
        model = _make_model(num_classes=2)
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with pytest.raises(ValueError, match="Invalid layer name"):
            _ = model.get_feature_maps(x, layer_names=["does_not_exist"])


class TestViTB16EmbeddingAndAttention:
    """Embedding and explicit attention rollout helpers."""

    def test_embedding_shape(self) -> None:
        """get_embedding returns (B, embedding_dim)."""
        model = _make_model(num_classes=5, global_pool="mean")
        model.eval()

        x = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            emb = model.get_embedding(x)

        assert emb.shape == (3, model.embedding_dim)

    def test_embedding_invalid_input_dim_raises(self) -> None:
        """get_embedding should validate input dimensionality."""
        model = _make_model(num_classes=2)

        bad_x = torch.randn(3, 224, 224)
        with pytest.raises(ValueError, match="Expected 4D input"):
            _ = model.get_embedding(bad_x)

    def test_get_attention_rollout(self) -> None:
        """get_attention_rollout returns a spatial relevance map."""
        model = _make_model(num_classes=3)
        model.eval()

        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            attn = model.get_attention_rollout(x)

        assert attn.ndim == 4  # (B, 1, H_p, W_p)
        assert attn.shape[0] == 2
        assert attn.shape[1] == 1

        # Values should be finite and non-negative
        assert torch.isfinite(attn).all()
        assert (attn >= 0).all()


class TestViTB16Shapes:
    """Static output shape calculations."""

    def test_get_layer_output_shapes_224(self) -> None:
        """Shapes for a 224×224 input."""
        model = _make_model(num_classes=2)
        shapes = model.get_layer_output_shapes((224, 224))

        assert "cls_token" in shapes
        assert "patch_tokens" in shapes
        assert "tokens" in shapes
        assert "patch_map" in shapes
        assert "attention_rollout" in shapes

        cls_shape = shapes["cls_token"]
        patch_tokens_shape = shapes["patch_tokens"]
        tokens_shape = shapes["tokens"]
        patch_map_shape = shapes["patch_map"]
        attn_shape = shapes["attention_rollout"]

        # Basic dimensionality checks
        assert cls_shape == (model.embedding_dim,)
        assert patch_tokens_shape[1] == model.embedding_dim
        assert tokens_shape[1] == model.embedding_dim

        num_patches = patch_tokens_shape[0]
        grid = int(math.sqrt(num_patches))
        assert grid * grid == num_patches

        # patch_map: (C, H_p, W_p)
        assert patch_map_shape[0] == model.embedding_dim
        assert patch_map_shape[1:] == (grid, grid)

        # attention_rollout: (1, H_p, W_p)
        assert attn_shape == (1, grid, grid)

    def test_get_layer_output_shapes_invalid_input(self) -> None:
        """Non-divisible image size should raise ValueError."""
        model = _make_model(num_classes=2)

        with pytest.raises(ValueError, match="not divisible by patch size"):
            _ = model.get_layer_output_shapes((230, 224))
