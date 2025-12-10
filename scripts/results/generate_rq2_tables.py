"""
Generate all RQ2 result tables in CSV and LaTeX format.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def format_mean_std(mean, std, decimals=3):
    """Format as mean ± std."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def generate_table5_stability():
    """Table 5: Explanation Stability Metrics"""

    # Load results
    results_df = pd.read_csv("results/rq2_complete/rq2_all_results.csv")

    # Create table
    table_data = []

    for model in ["baseline", "tri_objective"]:
        model_data = results_df[results_df["model"] == model]

        for seed in [42, 123, 456]:
            seed_data = model_data[model_data["seed"] == seed]
            if len(seed_data) == 0:
                continue

            row = seed_data.iloc[0]
            table_data.append(
                {
                    "Model": model.replace("_", " ").title(),
                    "Seed": seed,
                    "SSIM": f"{row['ssim']:.3f}",
                    "Rank Corr": f"{row['rank_corr']:.3f}",
                    "L2 Distance": f"{row['l2_distance']:.3f}",
                }
            )

        # Add mean row
        ssim_mean = model_data["ssim"].mean()
        ssim_std = model_data["ssim"].std()
        rank_mean = model_data["rank_corr"].mean()
        rank_std = model_data["rank_corr"].std()
        l2_mean = model_data["l2_distance"].mean()
        l2_std = model_data["l2_distance"].std()

        table_data.append(
            {
                "Model": model.replace("_", " ").title(),
                "Seed": "Mean",
                "SSIM": format_mean_std(ssim_mean, ssim_std),
                "Rank Corr": format_mean_std(rank_mean, rank_std),
                "L2 Distance": format_mean_std(l2_mean, l2_std),
            }
        )

    table_df = pd.DataFrame(table_data)

    # Save CSV
    output_dir = Path("results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    table_df.to_csv(output_dir / "table5_rq2_stability.csv", index=False)

    # Generate LaTeX
    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption="RQ2: Explanation Stability Metrics",
        label="tab:rq2_stability",
    )

    with open(output_dir / "table5_rq2_stability.tex", "w") as f:
        f.write(latex_str)

    print("✓ Table 5 generated")
    return table_df


def generate_table6_tcav():
    """Table 6: TCAV Concept Reliance Scores"""

    results_df = pd.read_csv("results/rq2_complete/rq2_all_results.csv")

    table_data = []

    for model in ["baseline", "tri_objective"]:
        model_data = results_df[results_df["model"] == model]

        for seed in [42, 123, 456]:
            seed_data = model_data[model_data["seed"] == seed]
            if len(seed_data) == 0:
                continue

            row = seed_data.iloc[0]
            table_data.append(
                {
                    "Model": model.replace("_", " ").title(),
                    "Seed": seed,
                    "Artifact TCAV": f"{row['artifact_mean']:.3f}",
                    "Medical TCAV": f"{row['medical_mean']:.3f}",
                    "Ratio": f"{row['tcav_ratio']:.2f}",
                }
            )

        # Mean
        artifact_mean = model_data["artifact_mean"].mean()
        artifact_std = model_data["artifact_mean"].std()
        medical_mean = model_data["medical_mean"].mean()
        medical_std = model_data["medical_mean"].std()
        ratio_mean = model_data["tcav_ratio"].mean()
        ratio_std = model_data["tcav_ratio"].std()

        table_data.append(
            {
                "Model": model.replace("_", " ").title(),
                "Seed": "Mean",
                "Artifact TCAV": format_mean_std(artifact_mean, artifact_std),
                "Medical TCAV": format_mean_std(medical_mean, medical_std),
                "Ratio": format_mean_std(ratio_mean, ratio_std, decimals=2),
            }
        )

    table_df = pd.DataFrame(table_data)

    output_dir = Path("results/tables")
    table_df.to_csv(output_dir / "table6_rq2_tcav.csv", index=False)

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption="RQ2: TCAV Concept Reliance Scores",
        label="tab:rq2_tcav",
    )

    with open(output_dir / "table6_rq2_tcav.tex", "w") as f:
        f.write(latex_str)

    print("✓ Table 6 generated")
    return table_df


def generate_table7_per_concept():
    """Table 7: Per-Concept TCAV Breakdown"""

    results_df = pd.read_csv("results/rq2_complete/rq2_all_results.csv")

    # Get concept columns
    artifact_concepts = ["ruler", "hair", "ink_marks", "black_borders"]
    medical_concepts = ["asymmetry", "pigment_network", "blue_white_veil"]

    table_data = []

    # Artifacts
    for concept in artifact_concepts:
        baseline_vals = results_df[results_df["model"] == "baseline"][concept]
        triobj_vals = results_df[results_df["model"] == "tri_objective"][concept]

        table_data.append(
            {
                "Concept": concept.replace("_", " ").title(),
                "Category": "Artifact",
                "Baseline TCAV": format_mean_std(
                    baseline_vals.mean(), baseline_vals.std()
                ),
                "Tri-objective TCAV": format_mean_std(
                    triobj_vals.mean(), triobj_vals.std()
                ),
            }
        )

    # Add artifact mean
    baseline_artifact = results_df[results_df["model"] == "baseline"]["artifact_mean"]
    triobj_artifact = results_df[results_df["model"] == "tri_objective"][
        "artifact_mean"
    ]

    table_data.append(
        {
            "Concept": "Mean (Artifacts)",
            "Category": "Artifact",
            "Baseline TCAV": format_mean_std(
                baseline_artifact.mean(), baseline_artifact.std()
            ),
            "Tri-objective TCAV": format_mean_std(
                triobj_artifact.mean(), triobj_artifact.std()
            ),
        }
    )

    # Medical
    for concept in medical_concepts:
        baseline_vals = results_df[results_df["model"] == "baseline"][concept]
        triobj_vals = results_df[results_df["model"] == "tri_objective"][concept]

        table_data.append(
            {
                "Concept": concept.replace("_", " ").title(),
                "Category": "Medical",
                "Baseline TCAV": format_mean_std(
                    baseline_vals.mean(), baseline_vals.std()
                ),
                "Tri-objective TCAV": format_mean_std(
                    triobj_vals.mean(), triobj_vals.std()
                ),
            }
        )

    # Medical mean
    baseline_medical = results_df[results_df["model"] == "baseline"]["medical_mean"]
    triobj_medical = results_df[results_df["model"] == "tri_objective"]["medical_mean"]

    table_data.append(
        {
            "Concept": "Mean (Medical)",
            "Category": "Medical",
            "Baseline TCAV": format_mean_std(
                baseline_medical.mean(), baseline_medical.std()
            ),
            "Tri-objective TCAV": format_mean_std(
                triobj_medical.mean(), triobj_medical.std()
            ),
        }
    )

    table_df = pd.DataFrame(table_data)

    output_dir = Path("results/tables")
    table_df.to_csv(output_dir / "table7_rq2_per_concept_tcav.csv", index=False)

    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        caption="RQ2: Per-Concept TCAV Breakdown",
        label="tab:rq2_per_concept",
    )

    with open(output_dir / "table7_rq2_per_concept_tcav.tex", "w") as f:
        f.write(latex_str)

    print("✓ Table 7 generated")
    return table_df


def main():
    """Generate all RQ2 tables."""

    print("\n" + "=" * 60)
    print("GENERATING RQ2 TABLES")
    print("=" * 60 + "\n")

    try:
        generate_table5_stability()
        generate_table6_tcav()
        generate_table7_per_concept()

        print("\n✓ All RQ2 tables generated!")
        print("Location: results/tables/")
    except FileNotFoundError:
        print("Error: RQ2 results file not found!")
        print("Please run evaluate_rq2_complete.py first.")
    except Exception as e:
        print(f"Error generating tables: {e}")


if __name__ == "__main__":
    main()
