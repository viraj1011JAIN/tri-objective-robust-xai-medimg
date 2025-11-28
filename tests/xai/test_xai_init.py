"""
Tests for src/xai/__init__.py module.

Tests the package initialization, conditional imports, and __all__ exports.

Author: Viraj Pankaj Jain
"""

import sys
from unittest.mock import patch

import pytest


class TestXAIModuleImports:
    """Test XAI module imports."""

    def test_gradcam_imports(self):
        """Test that Grad-CAM components are importable."""
        from src.xai import (
            GradCAM,
            GradCAMConfig,
            GradCAMPlusPlus,
            create_gradcam,
            get_recommended_layers,
        )

        assert GradCAM is not None
        assert GradCAMConfig is not None
        assert GradCAMPlusPlus is not None
        assert create_gradcam is not None
        assert get_recommended_layers is not None

    def test_stability_metrics_imports(self):
        """Test that stability metrics components are importable."""
        from src.xai import (
            SSIM,
            MultiScaleSSIM,
            StabilityMetrics,
            StabilityMetricsConfig,
            cosine_similarity,
            create_stability_metrics,
            normalized_l2_distance,
            spearman_correlation,
        )

        assert StabilityMetrics is not None
        assert StabilityMetricsConfig is not None
        assert create_stability_metrics is not None
        assert SSIM is not None
        assert MultiScaleSSIM is not None
        assert spearman_correlation is not None
        assert normalized_l2_distance is not None
        assert cosine_similarity is not None

    def test_faithfulness_imports(self):
        """Test that faithfulness metrics components are importable."""
        from src.xai import (
            DeletionMetric,
            FaithfulnessConfig,
            FaithfulnessMetrics,
            InsertionMetric,
            PointingGame,
            create_faithfulness_metrics,
        )

        assert FaithfulnessMetrics is not None
        assert FaithfulnessConfig is not None
        assert DeletionMetric is not None
        assert InsertionMetric is not None
        assert PointingGame is not None
        assert create_faithfulness_metrics is not None

    def test_baseline_quality_imports(self):
        """Test that baseline quality components are importable."""
        from src.xai import (
            BaselineExplanationQuality,
            BaselineQualityConfig,
            create_baseline_quality_evaluator,
        )

        assert BaselineExplanationQuality is not None
        assert BaselineQualityConfig is not None
        assert create_baseline_quality_evaluator is not None

    def test_concept_bank_imports(self):
        """Test that concept bank components are importable."""
        from src.xai import (
            CHEST_XRAY_ARTIFACT_CONCEPTS,
            CHEST_XRAY_MEDICAL_CONCEPTS,
            DERMOSCOPY_ARTIFACT_CONCEPTS,
            DERMOSCOPY_MEDICAL_CONCEPTS,
            ConceptBankConfig,
            ConceptBankCreator,
            create_concept_bank_creator,
        )

        assert ConceptBankCreator is not None
        assert ConceptBankConfig is not None
        assert create_concept_bank_creator is not None
        assert DERMOSCOPY_MEDICAL_CONCEPTS is not None
        assert DERMOSCOPY_ARTIFACT_CONCEPTS is not None
        assert CHEST_XRAY_MEDICAL_CONCEPTS is not None
        assert CHEST_XRAY_ARTIFACT_CONCEPTS is not None

    def test_tcav_imports(self):
        """Test that TCAV components are importable."""
        from src.xai import (
            TCAV,
            ActivationExtractor,
            CAVTrainer,
            ConceptDataset,
            TCAVConfig,
            create_tcav,
        )

        assert TCAV is not None
        assert TCAVConfig is not None
        assert ActivationExtractor is not None
        assert CAVTrainer is not None
        assert ConceptDataset is not None
        assert create_tcav is not None

    def test_baseline_tcav_imports(self):
        """Test that baseline TCAV components are importable."""
        from src.xai import (
            BaselineTCAVConfig,
            BaselineTCAVEvaluator,
            ConceptCategory,
            create_baseline_tcav_evaluator,
        )

        assert BaselineTCAVEvaluator is not None
        assert BaselineTCAVConfig is not None
        assert ConceptCategory is not None
        assert create_baseline_tcav_evaluator is not None

    def test_representation_analysis_imports(self):
        """Test that representation analysis components are importable."""
        from src.xai import (
            CKAAnalyzer,
            DomainGapAnalyzer,
            RepresentationConfig,
            SVCCAAnalyzer,
            create_cka_analyzer,
            create_domain_gap_analyzer,
            create_svcca_analyzer,
        )

        assert RepresentationConfig is not None
        assert CKAAnalyzer is not None
        assert SVCCAAnalyzer is not None
        assert DomainGapAnalyzer is not None
        assert create_cka_analyzer is not None
        assert create_svcca_analyzer is not None
        assert create_domain_gap_analyzer is not None

    def test_attention_rollout_conditional_import(self):
        """Test that AttentionRollout is conditionally imported."""
        from src import xai

        # Try to import - may be None if module not available
        attention_rollout = xai.AttentionRollout
        create_vit_explainer = xai.create_vit_explainer

        # Both should be None or both should exist
        if attention_rollout is None:
            assert create_vit_explainer is None
        else:
            assert create_vit_explainer is not None


class TestXAIModuleExports:
    """Test XAI module __all__ exports."""

    def test_all_contains_gradcam_exports(self):
        """Test that __all__ contains Grad-CAM exports."""
        from src import xai

        assert "GradCAM" in xai.__all__
        assert "GradCAMConfig" in xai.__all__
        assert "GradCAMPlusPlus" in xai.__all__
        assert "create_gradcam" in xai.__all__
        assert "get_recommended_layers" in xai.__all__

    def test_all_contains_vit_exports(self):
        """Test that __all__ contains ViT exports."""
        from src import xai

        assert "AttentionRollout" in xai.__all__
        assert "create_vit_explainer" in xai.__all__

    def test_all_contains_stability_exports(self):
        """Test that __all__ contains stability metrics exports."""
        from src import xai

        assert "StabilityMetrics" in xai.__all__
        assert "StabilityMetricsConfig" in xai.__all__
        assert "create_stability_metrics" in xai.__all__
        assert "SSIM" in xai.__all__
        assert "MultiScaleSSIM" in xai.__all__
        assert "spearman_correlation" in xai.__all__
        assert "normalized_l2_distance" in xai.__all__
        assert "cosine_similarity" in xai.__all__

    def test_all_contains_faithfulness_exports(self):
        """Test that __all__ contains faithfulness metrics exports."""
        from src import xai

        assert "FaithfulnessMetrics" in xai.__all__
        assert "FaithfulnessConfig" in xai.__all__
        assert "DeletionMetric" in xai.__all__
        assert "InsertionMetric" in xai.__all__
        assert "PointingGame" in xai.__all__
        assert "create_faithfulness_metrics" in xai.__all__

    def test_all_contains_baseline_quality_exports(self):
        """Test that __all__ contains baseline quality exports."""
        from src import xai

        assert "BaselineExplanationQuality" in xai.__all__
        assert "BaselineQualityConfig" in xai.__all__
        assert "create_baseline_quality_evaluator" in xai.__all__

    def test_all_contains_concept_bank_exports(self):
        """Test that __all__ contains concept bank exports."""
        from src import xai

        assert "ConceptBankCreator" in xai.__all__
        assert "ConceptBankConfig" in xai.__all__
        assert "create_concept_bank_creator" in xai.__all__
        assert "DERMOSCOPY_MEDICAL_CONCEPTS" in xai.__all__
        assert "DERMOSCOPY_ARTIFACT_CONCEPTS" in xai.__all__
        assert "CHEST_XRAY_MEDICAL_CONCEPTS" in xai.__all__
        assert "CHEST_XRAY_ARTIFACT_CONCEPTS" in xai.__all__

    def test_all_contains_tcav_exports(self):
        """Test that __all__ contains TCAV exports."""
        from src import xai

        assert "TCAV" in xai.__all__
        assert "TCAVConfig" in xai.__all__
        assert "ActivationExtractor" in xai.__all__
        assert "CAVTrainer" in xai.__all__
        assert "ConceptDataset" in xai.__all__
        assert "create_tcav" in xai.__all__

    def test_all_contains_baseline_tcav_exports(self):
        """Test that __all__ contains baseline TCAV exports."""
        from src import xai

        assert "BaselineTCAVEvaluator" in xai.__all__
        assert "BaselineTCAVConfig" in xai.__all__
        assert "ConceptCategory" in xai.__all__
        assert "create_baseline_tcav_evaluator" in xai.__all__

    def test_all_contains_representation_exports(self):
        """Test that __all__ contains representation analysis exports."""
        from src import xai

        assert "RepresentationConfig" in xai.__all__
        assert "CKAAnalyzer" in xai.__all__
        assert "SVCCAAnalyzer" in xai.__all__
        assert "DomainGapAnalyzer" in xai.__all__
        assert "create_cka_analyzer" in xai.__all__
        assert "create_svcca_analyzer" in xai.__all__
        assert "create_domain_gap_analyzer" in xai.__all__


class TestXAIModuleMetadata:
    """Test XAI module metadata."""

    def test_module_version(self):
        """Test that module has version attribute."""
        from src import xai

        assert hasattr(xai, "__version__")
        assert xai.__version__ == "6.8.0"

    def test_module_docstring(self):
        """Test that module has proper docstring."""
        from src import xai

        assert xai.__doc__ is not None
        assert "Explainable AI" in xai.__doc__
        assert "XAI" in xai.__doc__


class TestXAIConditionalImports:
    """Test conditional import behavior."""

    def test_attention_rollout_import_error_handling(self):
        """Test that ImportError for attention_rollout is handled gracefully."""
        # Remove module from sys.modules and force ImportError
        import sys

        # Save original module if it exists
        original_module = sys.modules.get("src.xai.attention_rollout")

        # Set to None to trigger ImportError on import attempt
        sys.modules["src.xai.attention_rollout"] = None

        try:
            # Force reimport of src.xai
            if "src.xai" in sys.modules:
                del sys.modules["src.xai"]

            import src.xai

            # Should not crash, and AttentionRollout should be None
            # This tests lines 85-87 (the except ImportError block)
            assert hasattr(src.xai, "AttentionRollout")
            assert hasattr(src.xai, "create_vit_explainer")

            # When import fails, both should be None
            if src.xai.AttentionRollout is None:
                assert src.xai.create_vit_explainer is None
        finally:
            # Restore original state
            if original_module is not None:
                sys.modules["src.xai.attention_rollout"] = original_module
            elif "src.xai.attention_rollout" in sys.modules:
                del sys.modules["src.xai.attention_rollout"]
            # Reimport to restore normal state
            if "src.xai" in sys.modules:
                del sys.modules["src.xai"]
            import src.xai

    def test_all_exports_are_accessible(self):
        """Test that all items in __all__ are accessible."""
        from src import xai

        for name in xai.__all__:
            obj = getattr(xai, name)
            # Object can be None for conditionally imported modules
            # but should still be accessible as an attribute
            assert hasattr(xai, name), f"{name} in __all__ but not accessible"

    def test_version_not_in_all(self):
        """Test that __version__ is not in __all__."""
        from src import xai

        assert "__version__" not in xai.__all__


class TestXAIModuleStructure:
    """Test the structure and organization of the XAI module."""

    def test_module_has_all_attribute(self):
        """Test that module has __all__ attribute."""
        from src import xai

        assert hasattr(xai, "__all__")
        assert isinstance(xai.__all__, list)
        assert len(xai.__all__) > 0

    def test_all_items_are_strings(self):
        """Test that all items in __all__ are strings."""
        from src import xai

        for item in xai.__all__:
            assert isinstance(item, str), f"Item {item} in __all__ is not a string"

    def test_no_duplicate_exports(self):
        """Test that there are no duplicate exports in __all__."""
        from src import xai

        assert len(xai.__all__) == len(set(xai.__all__)), "Duplicate exports in __all__"

    def test_reimport_stability(self):
        """Test that module can be reimported without issues."""
        import importlib

        import src.xai

        # Should be able to reload without errors
        importlib.reload(src.xai)

        # Basic checks still work
        assert hasattr(src.xai, "__all__")
        assert "GradCAM" in src.xai.__all__


class TestXAIModuleIntegration:
    """Test integration aspects of the XAI module."""

    def test_can_use_star_import(self):
        """Test that star import works correctly."""
        # This implicitly tests __all__
        namespace = {}
        exec("from src.xai import *", namespace)

        # Should have imported all items from __all__
        from src import xai

        for name in xai.__all__:
            # Skip None values (conditionally imported)
            if getattr(xai, name) is not None:
                assert name in namespace, f"{name} not imported with star import"

    def test_all_creator_functions_callable(self):
        """Test that all creator functions in __all__ are callable."""
        from src import xai

        creator_functions = [
            "create_gradcam",
            "create_vit_explainer",
            "create_stability_metrics",
            "create_faithfulness_metrics",
            "create_baseline_quality_evaluator",
            "create_concept_bank_creator",
            "create_tcav",
            "create_baseline_tcav_evaluator",
            "create_cka_analyzer",
            "create_svcca_analyzer",
            "create_domain_gap_analyzer",
        ]

        for func_name in creator_functions:
            func = getattr(xai, func_name)
            if func is not None:  # Skip conditionally imported functions
                assert callable(func), f"{func_name} is not callable"

    def test_concept_constants_are_not_none(self):
        """Test that concept constants are not None."""
        from src.xai import (
            CHEST_XRAY_ARTIFACT_CONCEPTS,
            CHEST_XRAY_MEDICAL_CONCEPTS,
            DERMOSCOPY_ARTIFACT_CONCEPTS,
            DERMOSCOPY_MEDICAL_CONCEPTS,
        )

        assert DERMOSCOPY_MEDICAL_CONCEPTS is not None
        assert DERMOSCOPY_ARTIFACT_CONCEPTS is not None
        assert CHEST_XRAY_MEDICAL_CONCEPTS is not None
        assert CHEST_XRAY_ARTIFACT_CONCEPTS is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
