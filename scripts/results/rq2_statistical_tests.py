"""
Statistical significance testing for RQ2 hypotheses.
"""

import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


class RQ2StatisticalTester:
    """Statistical testing for RQ2 hypotheses."""

    def __init__(self, results_path: str):
        self.results_df = pd.read_csv(results_path)
        self.baseline_data = self.results_df[self.results_df["model"] == "baseline"]
        self.triobj_data = self.results_df[self.results_df["model"] == "tri_objective"]

    def cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d

    def bootstrap_ci(self, data, n_bootstrap=10000, confidence=0.95):
        """Bootstrap confidence interval."""
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * (alpha / 2))
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return lower, upper

    def test_h2_1_stability(self) -> Dict:
        """H2.1: SSIM ≥ 0.75 (tri-objective significantly higher than baseline)"""

        baseline_ssim = self.baseline_data["ssim"].values
        triobj_ssim = self.triobj_data["ssim"].values

        # Paired t-test (higher SSIM is better)
        t_stat, p_value = stats.ttest_rel(
            triobj_ssim, baseline_ssim, alternative="greater"
        )

        # Effect size
        effect_size = self.cohens_d(triobj_ssim, baseline_ssim)

        # Confidence intervals
        triobj_ci = self.bootstrap_ci(triobj_ssim)
        baseline_ci = self.bootstrap_ci(baseline_ssim)

        # Check threshold
        triobj_mean = np.mean(triobj_ssim)
        threshold_met = triobj_mean >= 0.75

        result = {
            "hypothesis": "H2.1: SSIM ≥ 0.75",
            "baseline_mean": np.mean(baseline_ssim),
            "baseline_std": np.std(baseline_ssim, ddof=1),
            "baseline_ci": baseline_ci,
            "triobj_mean": triobj_mean,
            "triobj_std": np.std(triobj_ssim, ddof=1),
            "triobj_ci": triobj_ci,
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "threshold_met": threshold_met,
            "significant": p_value < 0.05,
            "supported": threshold_met and p_value < 0.05,
        }

        return result

    def test_h2_2_artifacts(self) -> Dict:
        """H2.2: Artifact TCAV ≤ 0.20 (tri-objective significantly lower than baseline)"""

        baseline_artifact = self.baseline_data["artifact_mean"].values
        triobj_artifact = self.triobj_data["artifact_mean"].values

        # Paired t-test (lower artifact reliance is better)
        t_stat, p_value = stats.ttest_rel(
            baseline_artifact, triobj_artifact, alternative="greater"
        )

        effect_size = self.cohens_d(baseline_artifact, triobj_artifact)

        triobj_ci = self.bootstrap_ci(triobj_artifact)
        baseline_ci = self.bootstrap_ci(baseline_artifact)

        triobj_mean = np.mean(triobj_artifact)
        threshold_met = triobj_mean <= 0.20

        result = {
            "hypothesis": "H2.2: Artifact TCAV ≤ 0.20",
            "baseline_mean": np.mean(baseline_artifact),
            "baseline_std": np.std(baseline_artifact, ddof=1),
            "baseline_ci": baseline_ci,
            "triobj_mean": triobj_mean,
            "triobj_std": np.std(triobj_artifact, ddof=1),
            "triobj_ci": triobj_ci,
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "threshold_met": threshold_met,
            "significant": p_value < 0.05,
            "supported": threshold_met and p_value < 0.05,
        }

        return result

    def test_h2_3_medical(self) -> Dict:
        """H2.3: Medical TCAV ≥ 0.65 (tri-objective significantly higher than baseline)"""

        baseline_medical = self.baseline_data["medical_mean"].values
        triobj_medical = self.triobj_data["medical_mean"].values

        # Paired t-test (higher medical reliance is better)
        t_stat, p_value = stats.ttest_rel(
            triobj_medical, baseline_medical, alternative="greater"
        )

        effect_size = self.cohens_d(triobj_medical, baseline_medical)

        triobj_ci = self.bootstrap_ci(triobj_medical)
        baseline_ci = self.bootstrap_ci(baseline_medical)

        triobj_mean = np.mean(triobj_medical)
        threshold_met = triobj_mean >= 0.65

        result = {
            "hypothesis": "H2.3: Medical TCAV ≥ 0.65",
            "baseline_mean": np.mean(baseline_medical),
            "baseline_std": np.std(baseline_medical, ddof=1),
            "baseline_ci": baseline_ci,
            "triobj_mean": triobj_mean,
            "triobj_std": np.std(triobj_medical, ddof=1),
            "triobj_ci": triobj_ci,
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "threshold_met": threshold_met,
            "significant": p_value < 0.05,
            "supported": threshold_met and p_value < 0.05,
        }

        return result

    def test_h2_4_ratio(self) -> Dict:
        """H2.4: TCAV Ratio ≥ 3.0 (tri-objective significantly higher than baseline)"""

        baseline_ratio = self.baseline_data["tcav_ratio"].values
        triobj_ratio = self.triobj_data["tcav_ratio"].values

        # Paired t-test (higher ratio is better)
        t_stat, p_value = stats.ttest_rel(
            triobj_ratio, baseline_ratio, alternative="greater"
        )

        effect_size = self.cohens_d(triobj_ratio, baseline_ratio)

        triobj_ci = self.bootstrap_ci(triobj_ratio)
        baseline_ci = self.bootstrap_ci(baseline_ratio)

        triobj_mean = np.mean(triobj_ratio)
        threshold_met = triobj_mean >= 3.0

        result = {
            "hypothesis": "H2.4: TCAV Ratio ≥ 3.0",
            "baseline_mean": np.mean(baseline_ratio),
            "baseline_std": np.std(baseline_ratio, ddof=1),
            "baseline_ci": baseline_ci,
            "triobj_mean": triobj_mean,
            "triobj_std": np.std(triobj_ratio, ddof=1),
            "triobj_ci": triobj_ci,
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "threshold_met": threshold_met,
            "significant": p_value < 0.05,
            "supported": threshold_met and p_value < 0.05,
        }

        return result

    def generate_results_table(self, results: Dict) -> pd.DataFrame:
        """Generate statistical results table."""

        table_data = []

        for result in results.values():
            table_data.append(
                {
                    "Hypothesis": result["hypothesis"],
                    "Baseline Mean": f"{result['baseline_mean']:.3f}",
                    "Baseline CI": f"[{result['baseline_ci'][0]:.3f}, {result['baseline_ci'][1]:.3f}]",
                    "Tri-objective Mean": f"{result['triobj_mean']:.3f}",
                    "Tri-objective CI": f"[{result['triobj_ci'][0]:.3f}, {result['triobj_ci'][1]:.3f}]",
                    "t-statistic": f"{result['t_statistic']:.3f}",
                    "p-value": (
                        f"{result['p_value']:.4f}"
                        if result["p_value"] >= 0.0001
                        else "< 0.0001"
                    ),
                    "Cohen's d": f"{result['effect_size']:.3f}",
                    "Threshold Met": "✓" if result["threshold_met"] else "✗",
                    "Significant": "✓" if result["significant"] else "✗",
                    "Supported": "✓" if result["supported"] else "✗",
                }
            )

        return pd.DataFrame(table_data)

    def run_all_tests(self) -> Dict:
        """Run all RQ2 hypothesis tests."""

        print("\n" + "=" * 60)
        print("RQ2 STATISTICAL HYPOTHESIS TESTING")
        print("=" * 60 + "\n")

        results = {}

        print("Testing H2.1: SSIM ≥ 0.75...")
        results["H2.1"] = self.test_h2_1_stability()

        print("Testing H2.2: Artifact TCAV ≤ 0.20...")
        results["H2.2"] = self.test_h2_2_artifacts()

        print("Testing H2.3: Medical TCAV ≥ 0.65...")
        results["H2.3"] = self.test_h2_3_medical()

        print("Testing H2.4: TCAV Ratio ≥ 3.0...")
        results["H2.4"] = self.test_h2_4_ratio()

        # Generate summary table
        results_df = self.generate_results_table(results)

        # Save results
        output_dir = Path("results/statistics")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df.to_csv(output_dir / "rq2_hypothesis_tests.csv", index=False)

        # LaTeX table
        latex_str = results_df.to_latex(
            index=False,
            escape=False,
            caption="RQ2: Statistical Hypothesis Testing Results",
            label="tab:rq2_statistics",
        )

        with open(output_dir / "rq2_hypothesis_tests.tex", "w") as f:
            f.write(latex_str)

        # Print summary
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)

        supported_count = 0
        for name, result in results.items():
            status = "SUPPORTED ✓" if result["supported"] else "NOT SUPPORTED ✗"
            print(f"{name}: {status}")
            if result["supported"]:
                supported_count += 1

        print(f"\nOverall: {supported_count}/4 hypotheses supported")
        print("Results saved to results/statistics/")

        return results


def main():
    """Main execution."""

    results_path = "results/rq2_complete/rq2_all_results.csv"

    try:
        tester = RQ2StatisticalTester(results_path)
        results = tester.run_all_tests()

    except FileNotFoundError:
        print(f"Error: Results file not found at {results_path}")
        print("Please run evaluate_rq2_complete.py first.")
    except Exception as e:
        print(f"Error in statistical testing: {e}")
        raise


if __name__ == "__main__":
    main()
