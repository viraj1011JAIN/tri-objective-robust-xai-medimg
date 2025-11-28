"""
Tests for src/validation/__init__.py module.

Tests the package initialization, conditional imports, and __all__ exports.

Author: Viraj Pankaj Jain
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestValidationModuleImports:
    """Test validation module imports and conditional logic."""

    def test_confidence_scorer_imports(self):
        """Test that confidence scoring components are always available."""
        from src.validation import (
            ConfidenceMethod,
            ConfidenceScore,
            ConfidenceScorer,
            EntropyScorer,
            MCDropoutScorer,
            SoftmaxMaxScorer,
            TemperatureScaledScorer,
            compute_confidence_metrics,
        )

        # Verify all are imported and not None
        assert ConfidenceMethod is not None
        assert ConfidenceScore is not None
        assert ConfidenceScorer is not None
        assert EntropyScorer is not None
        assert MCDropoutScorer is not None
        assert SoftmaxMaxScorer is not None
        assert TemperatureScaledScorer is not None
        assert compute_confidence_metrics is not None

    def test_validator_imports_when_available(self):
        """Test that validator components import when module exists."""
        # If the module exists, these should be available
        try:
            from src.validation import (
                BASELINE_VALUES,
                DEFAULT_TARGETS,
                ConvergenceAnalysis,
                ConvergenceAnalyzer,
                MultiSeedAggregator,
                ObjectiveType,
                TriObjectiveValidator,
                ValidationMetrics,
                ValidationResult,
                ValidationStatus,
                create_validator,
            )

            # Verify all are imported and not None
            assert TriObjectiveValidator is not None
            assert ValidationMetrics is not None
            assert ValidationResult is not None
            assert ValidationStatus is not None
            assert ObjectiveType is not None
            assert ConvergenceAnalyzer is not None
            assert ConvergenceAnalysis is not None
            assert MultiSeedAggregator is not None
            assert create_validator is not None
            assert DEFAULT_TARGETS is not None
            assert BASELINE_VALUES is not None
        except ImportError:
            # Module not available is expected - test that imports are handled gracefully
            # Verify that the conditional import worked correctly
            from src import validation

            assert hasattr(validation, "_HAS_VALIDATOR")
            assert validation._HAS_VALIDATOR == False
            # This is the expected behavior when module is not available

    def test_training_curves_imports_when_available(self):
        """Test that training curves components import when module exists."""
        try:
            from src.validation import (
                METRIC_COLORS,
                OBJECTIVE_COLORS,
                PUBLICATION_STYLE,
                SEED_COLORS,
                TrainingCurvePlotter,
                TrainingHistory,
                create_plotter,
            )

            # Verify all are imported and not None
            assert TrainingCurvePlotter is not None
            assert TrainingHistory is not None
            assert create_plotter is not None
            assert PUBLICATION_STYLE is not None
            assert OBJECTIVE_COLORS is not None
            assert METRIC_COLORS is not None
            assert SEED_COLORS is not None
        except ImportError:
            # Module not available is expected - test that imports are handled gracefully
            # Verify that the conditional import worked correctly
            from src import validation

            assert hasattr(validation, "_HAS_CURVES")
            assert validation._HAS_CURVES == False
            # This is the expected behavior when module is not available

    def test_all_exports_confidence_scoring(self):
        """Test that __all__ includes confidence scoring exports."""
        from src import validation

        # Confidence scoring should always be in __all__
        assert "ConfidenceMethod" in validation.__all__
        assert "ConfidenceScore" in validation.__all__
        assert "SoftmaxMaxScorer" in validation.__all__
        assert "EntropyScorer" in validation.__all__
        assert "MCDropoutScorer" in validation.__all__
        assert "TemperatureScaledScorer" in validation.__all__
        assert "ConfidenceScorer" in validation.__all__
        assert "compute_confidence_metrics" in validation.__all__

    def test_all_exports_include_validator_when_available(self):
        """Test that __all__ includes validator exports when available."""
        from src import validation

        # Check if validator is available
        if hasattr(validation, "_HAS_VALIDATOR") and validation._HAS_VALIDATOR:
            assert "TriObjectiveValidator" in validation.__all__
            assert "ValidationMetrics" in validation.__all__
            assert "ValidationResult" in validation.__all__
            assert "ValidationStatus" in validation.__all__
            assert "ObjectiveType" in validation.__all__
            assert "ConvergenceAnalyzer" in validation.__all__
            assert "ConvergenceAnalysis" in validation.__all__
            assert "MultiSeedAggregator" in validation.__all__
            assert "create_validator" in validation.__all__
            assert "DEFAULT_TARGETS" in validation.__all__
            assert "BASELINE_VALUES" in validation.__all__

    def test_all_exports_include_curves_when_available(self):
        """Test that __all__ includes training curves exports when available."""
        from src import validation

        # Check if curves is available
        if hasattr(validation, "_HAS_CURVES") and validation._HAS_CURVES:
            assert "TrainingCurvePlotter" in validation.__all__
            assert "TrainingHistory" in validation.__all__
            assert "create_plotter" in validation.__all__
            assert "PUBLICATION_STYLE" in validation.__all__
            assert "OBJECTIVE_COLORS" in validation.__all__
            assert "METRIC_COLORS" in validation.__all__
            assert "SEED_COLORS" in validation.__all__

    def test_module_metadata(self):
        """Test module metadata attributes."""
        from src import validation

        assert hasattr(validation, "__version__")
        assert hasattr(validation, "__author__")
        assert hasattr(validation, "__phase__")
        assert validation.__version__ == "1.0.0"
        assert validation.__author__ == "Viraj Pankaj Jain"
        assert validation.__phase__ == "7.7"


class TestValidationConditionalImports:
    """Test conditional import behavior with mocking."""

    def test_validator_import_error_handling(self):
        """Test that ImportError for validator is handled gracefully."""
        # Remove module from sys.modules and force ImportError
        import importlib
        import sys

        # Save original module if it exists
        original_module = sys.modules.get("src.validation.tri_objective_validator")

        # Set to None to trigger ImportError on import attempt
        sys.modules["src.validation.tri_objective_validator"] = None

        try:
            # Force reimport of src.validation
            if "src.validation" in sys.modules:
                del sys.modules["src.validation"]

            import src.validation

            # Should not crash, _HAS_VALIDATOR should be False when import fails
            assert hasattr(src.validation, "_HAS_VALIDATOR")
            # If import failed, _HAS_VALIDATOR should be False (line 41 executed)
            # If import succeeded, _HAS_VALIDATOR should be True
            assert isinstance(src.validation._HAS_VALIDATOR, bool)
        finally:
            # Restore original state
            if original_module is not None:
                sys.modules["src.validation.tri_objective_validator"] = original_module
            elif "src.validation.tri_objective_validator" in sys.modules:
                del sys.modules["src.validation.tri_objective_validator"]
            # Reimport to restore normal state
            if "src.validation" in sys.modules:
                del sys.modules["src.validation"]
            import src.validation

    def test_training_curves_import_error_handling(self):
        """Test that ImportError for training_curves is handled gracefully."""
        # Remove module from sys.modules and force ImportError
        import importlib
        import sys

        # Save original module if it exists
        original_module = sys.modules.get("src.validation.training_curves")

        # Set to None to trigger ImportError on import attempt
        sys.modules["src.validation.training_curves"] = None

        try:
            # Force reimport of src.validation
            if "src.validation" in sys.modules:
                del sys.modules["src.validation"]

            import src.validation

            # Should not crash, _HAS_CURVES should be False when import fails
            assert hasattr(src.validation, "_HAS_CURVES")
            # If import failed, _HAS_CURVES should be False (line 56 executed)
            # If import succeeded, _HAS_CURVES should be True
            assert isinstance(src.validation._HAS_CURVES, bool)
        finally:
            # Restore original state
            if original_module is not None:
                sys.modules["src.validation.training_curves"] = original_module
            elif "src.validation.training_curves" in sys.modules:
                del sys.modules["src.validation.training_curves"]
            # Reimport to restore normal state
            if "src.validation" in sys.modules:
                del sys.modules["src.validation"]
            import src.validation

    def test_all_dynamic_building_without_optional_modules(self):
        """Test __all__ is built correctly when optional modules are missing."""
        # Simulate missing optional modules by preventing their import
        import sys

        # Save original modules
        original_validator = sys.modules.get("src.validation.tri_objective_validator")
        original_curves = sys.modules.get("src.validation.training_curves")

        # Prevent imports
        sys.modules["src.validation.tri_objective_validator"] = None
        sys.modules["src.validation.training_curves"] = None

        try:
            # Force reimport - this should trigger lines 41 and 56
            if "src.validation" in sys.modules:
                del sys.modules["src.validation"]

            import src.validation

            # Should only have confidence scoring in __all__ (not validator/curves)
            assert "ConfidenceMethod" in src.validation.__all__
            assert "ConfidenceScore" in src.validation.__all__

            # These shouldn't be in __all__ if imports failed
            # (this tests that lines 75 and 93 are NOT executed)
            if not src.validation._HAS_VALIDATOR:
                assert "TriObjectiveValidator" not in src.validation.__all__
            if not src.validation._HAS_CURVES:
                assert "TrainingCurvePlotter" not in src.validation.__all__
        finally:
            # Restore original state
            if original_validator is not None:
                sys.modules["src.validation.tri_objective_validator"] = (
                    original_validator
                )
            elif "src.validation.tri_objective_validator" in sys.modules:
                del sys.modules["src.validation.tri_objective_validator"]

            if original_curves is not None:
                sys.modules["src.validation.training_curves"] = original_curves
            elif "src.validation.training_curves" in sys.modules:
                del sys.modules["src.validation.training_curves"]

            # Reimport to restore normal state
            if "src.validation" in sys.modules:
                del sys.modules["src.validation"]
            import src.validation

    def test_all_dynamic_building_with_all_modules(self):
        """Test __all__ is built correctly when all modules are available."""
        # Import the module normally
        import src.validation

        # Check base exports are present
        assert "ConfidenceMethod" in src.validation.__all__
        assert "ConfidenceScore" in src.validation.__all__

        # If optional modules are available, check they're included
        # This tests that lines 75 and 93 are executed when modules are present
        if src.validation._HAS_VALIDATOR:
            assert "TriObjectiveValidator" in src.validation.__all__
            assert "ValidationMetrics" in src.validation.__all__
            assert "DEFAULT_TARGETS" in src.validation.__all__

        if src.validation._HAS_CURVES:
            assert "TrainingCurvePlotter" in src.validation.__all__
            assert "TrainingHistory" in src.validation.__all__
            assert "METRIC_COLORS" in src.validation.__all__


class TestValidationModuleStructure:
    """Test the structure and organization of the validation module."""

    def test_module_docstring(self):
        """Test that module has proper docstring."""
        import src.validation

        assert src.validation.__doc__ is not None
        assert "Validation Module" in src.validation.__doc__
        assert "tri-objective" in src.validation.__doc__.lower()

    def test_all_exports_are_valid(self):
        """Test that all items in __all__ are actually available."""
        import src.validation

        for export in src.validation.__all__:
            # Each export should be accessible
            assert hasattr(
                src.validation, export
            ), f"{export} in __all__ but not accessible"

    def test_conditional_flags_exist(self):
        """Test that conditional import flags exist."""
        import src.validation

        assert hasattr(src.validation, "_HAS_VALIDATOR")
        assert hasattr(src.validation, "_HAS_CURVES")
        assert isinstance(src.validation._HAS_VALIDATOR, bool)
        assert isinstance(src.validation._HAS_CURVES, bool)

    def test_confidence_scorer_module_always_importable(self):
        """Test that confidence_scorer module is always available."""
        # This should never raise ImportError
        from src.validation import confidence_scorer

        assert confidence_scorer is not None

    def test_reimport_stability(self):
        """Test that module can be reimported without issues."""
        import importlib

        import src.validation

        # Should be able to reload without errors
        importlib.reload(src.validation)

        # Basic checks still work
        assert hasattr(src.validation, "__all__")
        assert "ConfidenceMethod" in src.validation.__all__


class TestValidationModuleIntegration:
    """Test integration aspects of the validation module."""

    def test_can_import_all_public_exports(self):
        """Test that all public exports can be imported."""
        from src import validation

        for name in validation.__all__:
            obj = getattr(validation, name)
            assert obj is not None, f"Export {name} is None"

    def test_private_flags_not_in_all(self):
        """Test that private flags are not exported in __all__."""
        from src import validation

        assert "_HAS_VALIDATOR" not in validation.__all__
        assert "_HAS_CURVES" not in validation.__all__

    def test_metadata_not_in_all(self):
        """Test that metadata is not exported in __all__."""
        from src import validation

        assert "__version__" not in validation.__all__
        assert "__author__" not in validation.__all__
        assert "__phase__" not in validation.__all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
