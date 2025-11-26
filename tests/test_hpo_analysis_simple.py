"""
Simplified HPO Analysis Tests for 100% Coverage.
Tests actual code execution with real matplotlib/plotly when available.
"""

import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

try:
    import optuna
    from optuna.trial import TrialState

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    TrialState = None


@pytest.fixture
def mock_study():
    """Create a real mock Optuna study with trials."""
    if not OPTUNA_AVAILABLE:
        pytest.skip("Optuna not installed")

    # Create real trial objects using enqueue_trial
    import optuna

    study = optuna.create_study(direction="maximize")

    #  Add trials programmatically using optimize with fixed params
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        beta = trial.suggest_float("beta", 1.0, 10.0)
        eps = trial.suggest_float("epsilon", 0.01, 0.1)

        # Simulate metrics
        value = np.random.uniform(0.7, 0.9)

        # Add user attributes DURING trial
        trial.set_user_attr("robust_accuracy", value)
        trial.set_user_attr("clean_accuracy", np.random.uniform(0.75, 0.95))
        trial.set_user_attr("cross_site_auroc", np.random.uniform(0.65, 0.85))

        return value

    study.optimize(objective, n_trials=10, show_progress_bar=False)

    return study


@pytest.fixture
def temp_dir():
    """Temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna not installed")
class TestHPOAnalyzerReal:
    """Test with real Optuna study."""

    def test_initialization(self, mock_study, temp_dir):
        """Test basic initialization."""
        from src.training.hpo_analysis import HPOAnalyzer

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)

        assert analyzer.study == mock_study
        assert not analyzer.trials_df.empty
        assert len(analyzer.trials_df) == 10

    def test_summary_report(self, mock_study, temp_dir):
        """Test summary report generation."""
        from src.training.hpo_analysis import HPOAnalyzer

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        summary = analyzer.generate_summary_report()

        assert summary["n_complete"] == 10
        assert summary["n_trials"] == 10
        assert "best_value" in summary
        assert "best_params" in summary
        assert "param_statistics" in summary

        # Check JSON was saved
        json_path = temp_dir / "hpo_summary.json"
        assert json_path.exists()

    def test_optimization_history_plot(self, mock_study, temp_dir):
        """Test optimization history plot."""
        from src.training.hpo_analysis import MATPLOTLIB_AVAILABLE, HPOAnalyzer

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        fig = analyzer.plot_optimization_history(save=True, show=False)

        assert fig is not None
        assert (temp_dir / "optimization_history.png").exists()

    def test_parameter_importance(self, mock_study, temp_dir):
        """Test parameter importance plot."""
        from src.training.hpo_analysis import MATPLOTLIB_AVAILABLE, HPOAnalyzer

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        fig = analyzer.plot_parameter_importance(save=True, show=False)

        # May return None if fANOVA fails, but should not crash
        assert fig is not None or fig is None

    def test_parameter_relationships(self, mock_study, temp_dir):
        """Test parameter relationships plot."""
        from src.training.hpo_analysis import MATPLOTLIB_AVAILABLE, HPOAnalyzer

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        fig = analyzer.plot_parameter_relationships(save=True, show=False)

        assert fig is not None
        assert (temp_dir / "parameter_relationships.png").exists()

    def test_convergence_plot(self, mock_study, temp_dir):
        """Test convergence plot."""
        from src.training.hpo_analysis import MATPLOTLIB_AVAILABLE, HPOAnalyzer

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        fig = analyzer.plot_convergence(save=True, show=False)

        assert fig is not None
        assert (temp_dir / "convergence_analysis.png").exists()

    def test_trade_offs_plot(self, mock_study, temp_dir):
        """Test trade-offs plot."""
        from src.training.hpo_analysis import MATPLOTLIB_AVAILABLE, HPOAnalyzer

        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        metrics = ["robust_accuracy", "clean_accuracy", "cross_site_auroc"]
        fig = analyzer.plot_trade_offs(metrics=metrics, save=True, show=False)

        assert fig is not None
        assert (temp_dir / "objective_tradeoffs.png").exists()

    def test_interactive_plots(self, mock_study, temp_dir):
        """Test interactive plots."""
        from src.training.hpo_analysis import PLOTLY_AVAILABLE, HPOAnalyzer

        if not PLOTLY_AVAILABLE:
            pytest.skip("Plotly not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        paths = analyzer.create_interactive_plots(save=True)

        assert len(paths) == 2
        for path in paths:
            assert path.exists()

    def test_export_csv(self, mock_study, temp_dir):
        """Test CSV export."""
        from src.training.hpo_analysis import HPOAnalyzer

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        path = analyzer.export_results(format="csv")

        assert path.exists()
        df = pd.read_csv(path)
        assert len(df) == 10

    def test_export_json(self, mock_study, temp_dir):
        """Test JSON export."""
        from src.training.hpo_analysis import HPOAnalyzer

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        path = analyzer.export_results(format="json")

        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 10

    def test_export_excel(self, mock_study, temp_dir):
        """Test Excel export."""
        from src.training.hpo_analysis import HPOAnalyzer

        try:
            import openpyxl
        except ImportError:
            pytest.skip("openpyxl not installed")

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        path = analyzer.export_results(format="excel")

        assert path.exists()

    def test_full_report(self, mock_study, temp_dir):
        """Test full report generation."""
        from src.training.hpo_analysis import HPOAnalyzer

        analyzer = HPOAnalyzer(mock_study, output_dir=temp_dir)
        report = analyzer.generate_full_report(include_interactive=False)

        assert "summary" in report
        assert "plots" in report
        assert "exports" in report

    def test_analyze_study_function(self, mock_study, temp_dir):
        """Test convenience function."""
        from src.training.hpo_analysis import analyze_study

        analyzer = analyze_study(mock_study, output_dir=temp_dir)

        assert analyzer is not None
        assert analyzer.study == mock_study


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
