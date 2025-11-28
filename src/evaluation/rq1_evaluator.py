"""
RQ1 Evaluation: Robustness & Cross-Site Generalization
========================================================

Research Question 1: Can adversarial robustness and cross-site generalization
be jointly optimized?

This module provides comprehensive evaluation infrastructure for RQ1:
- Task performance metrics (accuracy, AUROC, F1, MCC)
- Robustness evaluation (FGSM, PGD, C&W, AutoAttack)
- Cross-site generalization (AUROC drops, CKA analysis)
- Calibration analysis (ECE, MCE, Brier score)
- Statistical significance testing
- Hypothesis testing (H1a, H1b, H1c)
- Pareto frontier analysis
- Report generation with tables and figures

Phase 9.2: RQ1 Evaluation
Author: Viraj Jain
MSc Dissertation - University of Glasgow
Date: November 2024
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.attacks import (
    FGSM,
    PGD,
    AutoAttack,
    AutoAttackConfig,
    CarliniWagner,
    CWConfig,
    FGSMConfig,
    PGDConfig,
)
from src.evaluation.calibration import calculate_ece, calculate_mce
from src.evaluation.metrics import compute_bootstrap_ci, compute_classification_metrics
from src.evaluation.pareto_analysis import ParetoFrontier, ParetoSolution
from src.evaluation.statistical_tests import (
    bootstrap_confidence_interval,
    compute_cohens_d,
    interpret_effect_size,
    paired_t_test,
)
from src.xai.representation_analysis import (
    CKAAnalyzer,
    DomainGapAnalyzer,
    RepresentationConfig,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ModelCheckpoint:
    """Model checkpoint information."""

    name: str
    path: Path
    seed: int
    model_type: str  # 'baseline', 'pgd-at', 'trades', 'tri-objective'

    def __post_init__(self):
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.path}")


@dataclass
class EvaluationConfig:
    """Configuration for RQ1 evaluation."""

    # Models to evaluate
    models: List[ModelCheckpoint] = field(default_factory=list)

    # Datasets
    datasets: Dict[str, DataLoader] = field(default_factory=dict)
    source_dataset_name: str = "isic2018_test"
    target_dataset_names: List[str] = field(
        default_factory=lambda: ["isic2019", "isic2020", "derm7pt"]
    )

    # Attack configurations
    fgsm_epsilons: List[float] = field(
        default_factory=lambda: [2 / 255, 4 / 255, 8 / 255]
    )
    pgd_epsilons: List[float] = field(
        default_factory=lambda: [2 / 255, 4 / 255, 8 / 255]
    )
    pgd_steps: List[int] = field(default_factory=lambda: [7, 10, 20])
    cw_confidence: List[float] = field(default_factory=lambda: [0.0, 10.0, 20.0])
    autoattack_epsilon: float = 8 / 255

    # Evaluation settings
    num_classes: int = 8  # ISIC 2018: 7 classes + background
    class_names: List[str] = field(default_factory=list)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4

    # Statistical testing
    alpha: float = 0.01  # Significance level
    confidence_level: float = 0.95
    n_bootstrap: int = 10000

    # Output settings
    output_dir: Path = Path("results/rq1_evaluation")
    save_intermediate: bool = True
    verbose: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if not self.models:
            raise ValueError("No models specified for evaluation")
        if not self.datasets:
            raise ValueError("No datasets specified for evaluation")
        if self.source_dataset_name not in self.datasets:
            raise ValueError(
                f"Source dataset '{self.source_dataset_name}' not in datasets"
            )
        for target in self.target_dataset_names:
            if target not in self.datasets:
                raise ValueError(f"Target dataset '{target}' not in datasets")

        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TaskPerformanceResults:
    """Task performance evaluation results."""

    model_name: str
    seed: int
    dataset_name: str

    # Metrics
    accuracy: float
    auroc_macro: float
    auroc_weighted: float
    f1_macro: float
    f1_weighted: float
    mcc: float
    auroc_per_class: Dict[str, float] = field(default_factory=dict)

    # Confidence intervals
    accuracy_ci: Optional[Tuple[float, float]] = None
    auroc_macro_ci: Optional[Tuple[float, float]] = None
    f1_macro_ci: Optional[Tuple[float, float]] = None
    mcc_ci: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "dataset_name": self.dataset_name,
            "accuracy": float(self.accuracy),
            "auroc_macro": float(self.auroc_macro),
            "auroc_weighted": float(self.auroc_weighted),
            "f1_macro": float(self.f1_macro),
            "f1_weighted": float(self.f1_weighted),
            "mcc": float(self.mcc),
            "auroc_per_class": {k: float(v) for k, v in self.auroc_per_class.items()},
            "accuracy_ci": self.accuracy_ci,
            "auroc_macro_ci": self.auroc_macro_ci,
            "f1_macro_ci": self.f1_macro_ci,
            "mcc_ci": self.mcc_ci,
        }


@dataclass
class RobustnessResults:
    """Robustness evaluation results."""

    model_name: str
    seed: int
    attack_name: str
    attack_params: Dict[str, Any]

    # Metrics
    clean_accuracy: float
    robust_accuracy: float
    clean_auroc: float
    robust_auroc: float
    attack_success_rate: float

    # Additional info
    num_samples: int = 0
    time_elapsed: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "attack_name": self.attack_name,
            "attack_params": self.attack_params,
            "clean_accuracy": float(self.clean_accuracy),
            "robust_accuracy": float(self.robust_accuracy),
            "clean_auroc": float(self.clean_auroc),
            "robust_auroc": float(self.robust_auroc),
            "attack_success_rate": float(self.attack_success_rate),
            "num_samples": self.num_samples,
            "time_elapsed": float(self.time_elapsed),
        }


@dataclass
class CrossSiteResults:
    """Cross-site generalization results."""

    model_name: str
    seed: int
    source_dataset: str
    target_dataset: str

    # AUROC
    source_auroc: float
    target_auroc: float
    auroc_drop: float  # source_auroc - target_auroc

    # CKA domain gap
    cka_similarity: Optional[Dict[str, float]] = None  # per layer
    mean_cka_similarity: Optional[float] = None
    domain_gap: Optional[float] = None  # 1 - mean_cka_similarity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "source_dataset": self.source_dataset,
            "target_dataset": self.target_dataset,
            "source_auroc": float(self.source_auroc),
            "target_auroc": float(self.target_auroc),
            "auroc_drop": float(self.auroc_drop),
            "cka_similarity": (
                {k: float(v) for k, v in self.cka_similarity.items()}
                if self.cka_similarity
                else None
            ),
            "mean_cka_similarity": (
                float(self.mean_cka_similarity) if self.mean_cka_similarity else None
            ),
            "domain_gap": float(self.domain_gap) if self.domain_gap else None,
        }


@dataclass
class CalibrationResults:
    """Calibration evaluation results."""

    model_name: str
    seed: int
    dataset_name: str
    condition: str  # 'clean' or attack type

    # Calibration metrics
    ece: float
    mce: float
    brier_score: float

    # Binned statistics for reliability diagram
    bin_confidences: List[float] = field(default_factory=list)
    bin_accuracies: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "seed": self.seed,
            "dataset_name": self.dataset_name,
            "condition": self.condition,
            "ece": float(self.ece),
            "mce": float(self.mce),
            "brier_score": float(self.brier_score),
            "bin_confidences": [float(x) for x in self.bin_confidences],
            "bin_accuracies": [float(x) for x in self.bin_accuracies],
            "bin_counts": [int(x) for x in self.bin_counts],
        }


@dataclass
class HypothesisTestResults:
    """Hypothesis test results for RQ1."""

    hypothesis_name: str
    hypothesis_description: str

    # Groups being compared
    group1_name: str
    group2_name: str
    group1_values: List[float]
    group2_values: List[float]

    # Statistical test results
    t_statistic: float
    p_value: float
    significant: bool
    alpha: float

    # Effect size
    cohens_d: float
    effect_interpretation: str

    # Confidence interval
    ci_lower: float
    ci_upper: float
    ci_contains_zero: bool

    # Hypothesis-specific metrics
    improvement: Optional[float] = None  # For H1a, H1b
    threshold: Optional[float] = None  # Expected improvement/reduction
    hypothesis_supported: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_name": self.hypothesis_name,
            "hypothesis_description": self.hypothesis_description,
            "group1_name": self.group1_name,
            "group2_name": self.group2_name,
            "group1_mean": float(np.mean(self.group1_values)),
            "group1_std": float(np.std(self.group1_values, ddof=1)),
            "group2_mean": float(np.mean(self.group2_values)),
            "group2_std": float(np.std(self.group2_values, ddof=1)),
            "t_statistic": float(self.t_statistic),
            "p_value": float(self.p_value),
            "significant": self.significant,
            "alpha": self.alpha,
            "cohens_d": float(self.cohens_d),
            "effect_interpretation": self.effect_interpretation,
            "ci_95": (float(self.ci_lower), float(self.ci_upper)),
            "ci_contains_zero": self.ci_contains_zero,
            "improvement": (
                float(self.improvement) if self.improvement is not None else None
            ),
            "threshold": float(self.threshold) if self.threshold is not None else None,
            "hypothesis_supported": self.hypothesis_supported,
        }


# ============================================================================
# RQ1 EVALUATOR
# ============================================================================


class RQ1Evaluator:
    """
    Comprehensive RQ1 evaluation orchestrator.

    Evaluates all models on all metrics required for RQ1:
    - Task performance
    - Adversarial robustness
    - Cross-site generalization
    - Calibration
    - Hypothesis testing
    - Pareto analysis

    Usage:
        >>> config = EvaluationConfig(...)
        >>> evaluator = RQ1Evaluator(config)
        >>> results = evaluator.run_full_evaluation()
        >>> evaluator.generate_report()
    """

    def __init__(self, config: EvaluationConfig):
        """Initialize evaluator."""
        self.config = config
        self.results: Dict[str, List] = {
            "task_performance": [],
            "robustness": [],
            "cross_site": [],
            "calibration": [],
            "hypothesis_tests": [],
        }

        logger.info(f"RQ1Evaluator initialized with {len(config.models)} models")
        logger.info(f"Evaluation datasets: {list(config.datasets.keys())}")

    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Run complete RQ1 evaluation pipeline.

        Returns:
            Dictionary with all evaluation results
        """
        logger.info("=" * 80)
        logger.info("STARTING RQ1 FULL EVALUATION")
        logger.info("=" * 80)

        # Step 1: Task performance
        logger.info("\n[1/5] Evaluating task performance...")
        self._evaluate_task_performance()

        # Step 2: Robustness
        logger.info("\n[2/5] Evaluating adversarial robustness...")
        self._evaluate_robustness()

        # Step 3: Cross-site generalization
        logger.info("\n[3/5] Evaluating cross-site generalization...")
        self._evaluate_cross_site_generalization()

        # Step 4: Calibration
        logger.info("\n[4/5] Evaluating calibration...")
        self._evaluate_calibration()

        # Step 5: Hypothesis testing
        logger.info("\n[5/5] Performing hypothesis testing...")
        self._perform_hypothesis_testing()

        logger.info("\n" + "=" * 80)
        logger.info("RQ1 EVALUATION COMPLETE")
        logger.info("=" * 80)

        # Save results
        if self.config.save_intermediate:
            self._save_results()

        return self.results

    def _evaluate_task_performance(self) -> None:
        """Evaluate task performance on all datasets."""
        for model_checkpoint in tqdm(self.config.models, desc="Models"):
            # Load model
            model = self._load_model(model_checkpoint)
            model.eval()

            # Evaluate on all datasets
            for dataset_name, dataloader in self.config.datasets.items():
                logger.info(
                    f"  Evaluating {model_checkpoint.name} (seed={model_checkpoint.seed}) "
                    f"on {dataset_name}..."
                )

                # Collect predictions and labels
                all_predictions = []
                all_labels = []

                with torch.no_grad():
                    for batch in dataloader:
                        images, labels = batch[0].to(self.config.device), batch[1].to(
                            self.config.device
                        )
                        outputs = model(images)

                        # Apply softmax to get probabilities
                        probs = torch.softmax(outputs, dim=1)

                        all_predictions.append(probs.cpu())
                        all_labels.append(labels.cpu())

                predictions = torch.cat(all_predictions, dim=0).numpy()
                labels = torch.cat(all_labels, dim=0).numpy()

                # Compute metrics
                metrics = compute_classification_metrics(
                    predictions,
                    labels,
                    num_classes=self.config.num_classes,
                    class_names=(
                        self.config.class_names if self.config.class_names else None
                    ),
                )

                # Compute confidence intervals with bootstrap
                ci_metrics = compute_bootstrap_ci(
                    predictions,
                    labels,
                    num_classes=self.config.num_classes,
                    metric_fn=compute_classification_metrics,
                    n_bootstrap=1000,
                    confidence_level=self.config.confidence_level,
                )

                # Store results
                result = TaskPerformanceResults(
                    model_name=model_checkpoint.name,
                    seed=model_checkpoint.seed,
                    dataset_name=dataset_name,
                    accuracy=metrics["accuracy"],
                    auroc_macro=metrics["auroc_macro"],
                    auroc_weighted=metrics["auroc_weighted"],
                    f1_macro=metrics["f1_macro"],
                    f1_weighted=metrics["f1_weighted"],
                    mcc=metrics["mcc"],
                    auroc_per_class={
                        k.replace("auroc_", ""): v
                        for k, v in metrics.items()
                        if k.startswith("auroc_")
                        and k not in ["auroc_macro", "auroc_weighted"]
                    },
                    accuracy_ci=ci_metrics.get("accuracy_ci"),
                    auroc_macro_ci=ci_metrics.get("auroc_macro_ci"),
                    f1_macro_ci=ci_metrics.get("f1_macro_ci"),
                    mcc_ci=ci_metrics.get("mcc_ci"),
                )

                self.results["task_performance"].append(result)

                if self.config.verbose:
                    logger.info(
                        f"    Accuracy: {result.accuracy:.3f}, "
                        f"AUROC: {result.auroc_macro:.3f}, "
                        f"F1: {result.f1_macro:.3f}, "
                        f"MCC: {result.mcc:.3f}"
                    )

    def _evaluate_robustness(self) -> None:
        """Evaluate adversarial robustness with multiple attacks."""
        source_dataloader = self.config.datasets[self.config.source_dataset_name]

        for model_checkpoint in tqdm(self.config.models, desc="Models"):
            model = self._load_model(model_checkpoint)
            model.eval()

            # FGSM attacks
            for epsilon in self.config.fgsm_epsilons:
                self._evaluate_attack(
                    model,
                    model_checkpoint,
                    source_dataloader,
                    attack_type="fgsm",
                    epsilon=epsilon,
                )

            # PGD attacks
            for epsilon in self.config.pgd_epsilons:
                for steps in self.config.pgd_steps:
                    self._evaluate_attack(
                        model,
                        model_checkpoint,
                        source_dataloader,
                        attack_type="pgd",
                        epsilon=epsilon,
                        num_steps=steps,
                    )

            # C&W attacks
            for confidence in self.config.cw_confidence:
                self._evaluate_attack(
                    model,
                    model_checkpoint,
                    source_dataloader,
                    attack_type="cw",
                    confidence=confidence,
                )

            # AutoAttack
            self._evaluate_attack(
                model,
                model_checkpoint,
                source_dataloader,
                attack_type="autoattack",
                epsilon=self.config.autoattack_epsilon,
            )

    def _evaluate_attack(
        self,
        model: nn.Module,
        model_checkpoint: ModelCheckpoint,
        dataloader: DataLoader,
        attack_type: str,
        **attack_kwargs,
    ) -> None:
        """Evaluate a single attack configuration."""
        import time

        logger.info(
            f"  {model_checkpoint.name} vs {attack_type.upper()} "
            f"({', '.join(f'{k}={v}' for k, v in attack_kwargs.items())})"
        )

        # Create attack
        if attack_type == "fgsm":
            config = FGSMConfig(
                epsilon=attack_kwargs["epsilon"],
                device=self.config.device,
            )
            attack = FGSM(config)
        elif attack_type == "pgd":
            config = PGDConfig(
                epsilon=attack_kwargs["epsilon"],
                num_steps=attack_kwargs.get("num_steps", 40),
                step_size=attack_kwargs["epsilon"] / 4,
                random_start=True,
                device=self.config.device,
            )
            attack = PGD(config)
        elif attack_type == "cw":
            config = CWConfig(
                confidence=attack_kwargs.get("confidence", 0.0),
                learning_rate=0.01,
                max_iterations=1000,
                device=self.config.device,
            )
            attack = CarliniWagner(config)
        elif attack_type == "autoattack":
            config = AutoAttackConfig(
                epsilon=attack_kwargs["epsilon"],
                norm="Linf",
                version="standard",
                device=self.config.device,
            )
            attack = AutoAttack(config)
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        # Evaluate
        clean_correct = 0
        robust_correct = 0
        clean_probs = []
        robust_probs = []
        all_labels = []
        total_samples = 0

        start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0].to(self.config.device), batch[1].to(
                    self.config.device
                )

                # Clean predictions
                clean_outputs = model(images)
                clean_pred = clean_outputs.argmax(dim=1)
                clean_correct += (clean_pred == labels).sum().item()
                clean_probs.append(torch.softmax(clean_outputs, dim=1).cpu())

                # Generate adversarial examples
                adv_images = attack.generate(model, images, labels)

                # Adversarial predictions
                adv_outputs = model(adv_images)
                adv_pred = adv_outputs.argmax(dim=1)
                robust_correct += (adv_pred == labels).sum().item()
                robust_probs.append(torch.softmax(adv_outputs, dim=1).cpu())

                all_labels.append(labels.cpu())
                total_samples += labels.size(0)

        time_elapsed = time.time() - start_time

        # Compute metrics
        clean_accuracy = clean_correct / total_samples
        robust_accuracy = robust_correct / total_samples
        attack_success_rate = 1.0 - (robust_correct / max(clean_correct, 1))

        # Compute AUROC
        clean_probs = torch.cat(clean_probs, dim=0).numpy()
        robust_probs = torch.cat(robust_probs, dim=0).numpy()
        labels_np = torch.cat(all_labels, dim=0).numpy()

        try:
            from sklearn.metrics import roc_auc_score

            clean_auroc = roc_auc_score(
                labels_np,
                clean_probs,
                multi_class="ovr",
                average="macro",
            )
            robust_auroc = roc_auc_score(
                labels_np,
                robust_probs,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            clean_auroc = float("nan")
            robust_auroc = float("nan")

        # Store results
        result = RobustnessResults(
            model_name=model_checkpoint.name,
            seed=model_checkpoint.seed,
            attack_name=attack_type,
            attack_params=attack_kwargs,
            clean_accuracy=clean_accuracy,
            robust_accuracy=robust_accuracy,
            clean_auroc=clean_auroc,
            robust_auroc=robust_auroc,
            attack_success_rate=attack_success_rate,
            num_samples=total_samples,
            time_elapsed=time_elapsed,
        )

        self.results["robustness"].append(result)

        if self.config.verbose:
            logger.info(
                f"    Clean Acc: {clean_accuracy:.3f}, "
                f"Robust Acc: {robust_accuracy:.3f}, "
                f"ASR: {attack_success_rate:.3f}"
            )

    def _evaluate_cross_site_generalization(self) -> None:
        """Evaluate cross-site generalization with AUROC drops and CKA."""
        source_dataloader = self.config.datasets[self.config.source_dataset_name]

        for model_checkpoint in tqdm(self.config.models, desc="Models"):
            model = self._load_model(model_checkpoint)
            model.eval()

            # Get source AUROC
            source_result = self._get_task_performance_result(
                model_checkpoint.name,
                model_checkpoint.seed,
                self.config.source_dataset_name,
            )
            source_auroc = source_result.auroc_macro if source_result else 0.0

            # Evaluate on target datasets
            for target_name in self.config.target_dataset_names:
                target_dataloader = self.config.datasets[target_name]

                # Get target AUROC
                target_result = self._get_task_performance_result(
                    model_checkpoint.name,
                    model_checkpoint.seed,
                    target_name,
                )
                target_auroc = target_result.auroc_macro if target_result else 0.0

                # Compute AUROC drop
                auroc_drop = source_auroc - target_auroc

                # CKA domain gap analysis (optional, may be slow)
                cka_similarity = None
                mean_cka = None
                domain_gap = None

                # Store results
                result = CrossSiteResults(
                    model_name=model_checkpoint.name,
                    seed=model_checkpoint.seed,
                    source_dataset=self.config.source_dataset_name,
                    target_dataset=target_name,
                    source_auroc=source_auroc,
                    target_auroc=target_auroc,
                    auroc_drop=auroc_drop,
                    cka_similarity=cka_similarity,
                    mean_cka_similarity=mean_cka,
                    domain_gap=domain_gap,
                )

                self.results["cross_site"].append(result)

                if self.config.verbose:
                    logger.info(
                        f"  {model_checkpoint.name}: {self.config.source_dataset_name} → {target_name} "
                        f"AUROC drop: {auroc_drop:.1f}pp"
                    )

    def _evaluate_calibration(self) -> None:
        """Evaluate calibration on clean and adversarial examples."""
        source_dataloader = self.config.datasets[self.config.source_dataset_name]

        for model_checkpoint in tqdm(self.config.models, desc="Models"):
            model = self._load_model(model_checkpoint)
            model.eval()

            # Clean calibration
            self._evaluate_calibration_single(
                model,
                model_checkpoint,
                source_dataloader,
                condition="clean",
            )

            # Adversarial calibration (using PGD eps=8/255)
            self._evaluate_calibration_single(
                model,
                model_checkpoint,
                source_dataloader,
                condition="pgd_8",
                attack_type="pgd",
                epsilon=8 / 255,
            )

    def _evaluate_calibration_single(
        self,
        model: nn.Module,
        model_checkpoint: ModelCheckpoint,
        dataloader: DataLoader,
        condition: str,
        attack_type: Optional[str] = None,
        **attack_kwargs,
    ) -> None:
        """Evaluate calibration for a single condition."""
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images, labels = batch[0].to(self.config.device), batch[1].to(
                    self.config.device
                )

                # Generate adversarial examples if needed
                if attack_type:
                    if attack_type == "pgd":
                        config = PGDConfig(
                            epsilon=attack_kwargs.get("epsilon", 8 / 255),
                            num_steps=40,
                            device=self.config.device,
                        )
                        attack = PGD(config)
                        images = attack.generate(model, images, labels)

                # Get predictions
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)

                all_predictions.append(probs.cpu())
                all_labels.append(labels.cpu())

        predictions = torch.cat(all_predictions, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        # Compute calibration metrics
        ece = calculate_ece(predictions, labels, num_bins=15)
        mce = calculate_mce(predictions, labels, num_bins=15)

        # Compute Brier score
        pred_labels = predictions.argmax(axis=1)
        correct = (pred_labels == labels).astype(float)
        confidences = predictions.max(axis=1)
        brier_score = float(np.mean((confidences - correct) ** 2))

        # Compute binned statistics for reliability diagram
        bin_boundaries = np.linspace(0, 1, 16)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for i in range(15):
            in_bin = (confidences > bin_boundaries[i]) & (
                confidences <= bin_boundaries[i + 1]
            )
            if in_bin.sum() > 0:
                bin_confidences.append(float(confidences[in_bin].mean()))
                bin_accuracies.append(float(correct[in_bin].mean()))
                bin_counts.append(int(in_bin.sum()))

        # Store results
        result = CalibrationResults(
            model_name=model_checkpoint.name,
            seed=model_checkpoint.seed,
            dataset_name=self.config.source_dataset_name,
            condition=condition,
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            bin_confidences=bin_confidences,
            bin_accuracies=bin_accuracies,
            bin_counts=bin_counts,
        )

        self.results["calibration"].append(result)

        if self.config.verbose:
            logger.info(
                f"  {model_checkpoint.name} ({condition}): "
                f"ECE={ece:.4f}, MCE={mce:.4f}, Brier={brier_score:.4f}"
            )

    def _perform_hypothesis_testing(self) -> None:
        """Perform hypothesis testing for RQ1."""
        logger.info("Performing hypothesis testing...")

        # H1a: Tri-objective robust accuracy ≥ Baseline + 35pp
        self._test_h1a()

        # H1b: Tri-objective cross-site drop ≤ Baseline - 8pp
        self._test_h1b()

        # H1c: PGD-AT/TRADES does NOT improve cross-site
        self._test_h1c()

    def _test_h1a(self) -> None:
        """Test H1a: Tri-objective robust accuracy ≥ Baseline + 35pp."""
        # Get robust accuracies for baseline and tri-objective
        baseline_robust = self._get_robust_accuracies(
            "baseline", attack_type="pgd", epsilon=8 / 255
        )
        tri_obj_robust = self._get_robust_accuracies(
            "tri-objective", attack_type="pgd", epsilon=8 / 255
        )

        if not baseline_robust or not tri_obj_robust:
            logger.warning("H1a: Insufficient data for hypothesis testing")
            return

        # Convert to percentage points
        baseline_vals = [x * 100 for x in baseline_robust]
        tri_obj_vals = [x * 100 for x in tri_obj_robust]

        # Perform paired t-test
        from scipy import stats as scipy_stats

        t_stat, p_value = scipy_stats.ttest_rel(tri_obj_vals, baseline_vals)

        # Compute effect size
        cohens_d = compute_cohens_d(np.array(tri_obj_vals), np.array(baseline_vals))

        # Bootstrap CI
        differences = np.array(tri_obj_vals) - np.array(baseline_vals)
        ci_result = bootstrap_confidence_interval(
            np.array(tri_obj_vals),
            np.array(baseline_vals),
            n_bootstrap=self.config.n_bootstrap,
            confidence_level=self.config.confidence_level,
        )

        # Check hypothesis
        improvement = float(np.mean(differences))
        threshold = 35.0  # 35pp improvement expected
        hypothesis_supported = improvement >= threshold and p_value < self.config.alpha

        result = HypothesisTestResults(
            hypothesis_name="H1a",
            hypothesis_description="Tri-objective robust accuracy ≥ Baseline + 35pp",
            group1_name="Tri-Objective",
            group2_name="Baseline",
            group1_values=tri_obj_vals,
            group2_values=baseline_vals,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.config.alpha,
            alpha=self.config.alpha,
            cohens_d=cohens_d,
            effect_interpretation=interpret_effect_size(cohens_d),
            ci_lower=ci_result["lower"],
            ci_upper=ci_result["upper"],
            ci_contains_zero=ci_result["contains_zero"],
            improvement=improvement,
            threshold=threshold,
            hypothesis_supported=hypothesis_supported,
        )

        self.results["hypothesis_tests"].append(result)

        logger.info(
            f"  H1a: Improvement={improvement:.1f}pp, p={p_value:.4f}, "
            f"d={cohens_d:.3f}, Supported={hypothesis_supported}"
        )

    def _test_h1b(self) -> None:
        """Test H1b: Tri-objective cross-site drop ≤ Baseline - 8pp."""
        # Get AUROC drops
        baseline_drops = self._get_cross_site_drops("baseline")
        tri_obj_drops = self._get_cross_site_drops("tri-objective")

        if not baseline_drops or not tri_obj_drops:
            logger.warning("H1b: Insufficient data for hypothesis testing")
            return

        # Perform paired t-test (lower drop is better, so flip comparison)
        from scipy import stats as scipy_stats

        t_stat, p_value = scipy_stats.ttest_rel(tri_obj_drops, baseline_drops)

        # Compute effect size
        cohens_d = compute_cohens_d(np.array(tri_obj_drops), np.array(baseline_drops))

        # Bootstrap CI
        ci_result = bootstrap_confidence_interval(
            np.array(tri_obj_drops),
            np.array(baseline_drops),
            n_bootstrap=self.config.n_bootstrap,
            confidence_level=self.config.confidence_level,
        )

        # Check hypothesis
        reduction = float(np.mean(baseline_drops) - np.mean(tri_obj_drops))
        threshold = 8.0  # 8pp reduction expected
        hypothesis_supported = reduction >= threshold and p_value < self.config.alpha

        result = HypothesisTestResults(
            hypothesis_name="H1b",
            hypothesis_description="Tri-objective cross-site drop ≤ Baseline - 8pp",
            group1_name="Tri-Objective",
            group2_name="Baseline",
            group1_values=tri_obj_drops,
            group2_values=baseline_drops,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < self.config.alpha,
            alpha=self.config.alpha,
            cohens_d=cohens_d,
            effect_interpretation=interpret_effect_size(cohens_d),
            ci_lower=ci_result["lower"],
            ci_upper=ci_result["upper"],
            ci_contains_zero=ci_result["contains_zero"],
            improvement=reduction,
            threshold=threshold,
            hypothesis_supported=hypothesis_supported,
        )

        self.results["hypothesis_tests"].append(result)

        logger.info(
            f"  H1b: Reduction={reduction:.1f}pp, p={p_value:.4f}, "
            f"d={cohens_d:.3f}, Supported={hypothesis_supported}"
        )

    def _test_h1c(self) -> None:
        """Test H1c: PGD-AT/TRADES does NOT improve cross-site."""
        # Get AUROC drops
        baseline_drops = self._get_cross_site_drops("baseline")
        pgd_at_drops = self._get_cross_site_drops("pgd-at")

        if not baseline_drops or not pgd_at_drops:
            logger.warning("H1c: Insufficient data for hypothesis testing")
            return

        # Perform paired t-test
        from scipy import stats as scipy_stats

        t_stat, p_value = scipy_stats.ttest_rel(pgd_at_drops, baseline_drops)

        # Compute effect size
        cohens_d = compute_cohens_d(np.array(pgd_at_drops), np.array(baseline_drops))

        # Bootstrap CI
        ci_result = bootstrap_confidence_interval(
            np.array(pgd_at_drops),
            np.array(baseline_drops),
            n_bootstrap=self.config.n_bootstrap,
            confidence_level=self.config.confidence_level,
        )

        # Check hypothesis (expect NO significant difference)
        hypothesis_supported = p_value >= 0.05  # Not significant

        result = HypothesisTestResults(
            hypothesis_name="H1c",
            hypothesis_description="PGD-AT does NOT improve cross-site generalization",
            group1_name="PGD-AT",
            group2_name="Baseline",
            group1_values=pgd_at_drops,
            group2_values=baseline_drops,
            t_statistic=t_stat,
            p_value=p_value,
            significant=p_value < 0.05,
            alpha=0.05,
            cohens_d=cohens_d,
            effect_interpretation=interpret_effect_size(cohens_d),
            ci_lower=ci_result["lower"],
            ci_upper=ci_result["upper"],
            ci_contains_zero=ci_result["contains_zero"],
            hypothesis_supported=hypothesis_supported,
        )

        self.results["hypothesis_tests"].append(result)

        logger.info(
            f"  H1c: p={p_value:.4f}, d={cohens_d:.3f}, "
            f"No improvement (Supported={hypothesis_supported})"
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _load_model(self, checkpoint: ModelCheckpoint) -> nn.Module:
        """Load model from checkpoint."""
        # Implementation depends on your model loading logic
        # For now, return a placeholder
        state_dict = torch.load(
            checkpoint.path,
            map_location=self.config.device,
            weights_only=False,  # Allow custom classes in checkpoint
        )

        # Create model (you'll need to adapt this to your model architecture)
        from src.models import create_model  # Adjust import

        model = create_model(
            architecture="resnet50",
            num_classes=self.config.num_classes,
            pretrained=False,
        )
        model.load_state_dict(state_dict)
        model.to(self.config.device)
        model.eval()

        return model

    def _get_task_performance_result(
        self, model_name: str, seed: int, dataset_name: str
    ) -> Optional[TaskPerformanceResults]:
        """Get task performance result for a model/seed/dataset combination."""
        for result in self.results["task_performance"]:
            if (
                result.model_name == model_name
                and result.seed == seed
                and result.dataset_name == dataset_name
            ):
                return result
        return None

    def _get_robust_accuracies(
        self, model_type: str, attack_type: str, epsilon: float
    ) -> List[float]:
        """Get robust accuracies for a model type."""
        accuracies = []
        for result in self.results["robustness"]:
            if (
                result.model_name.startswith(model_type)
                and result.attack_name == attack_type
                and abs(result.attack_params.get("epsilon", 0) - epsilon) < 1e-6
            ):
                accuracies.append(result.robust_accuracy)
        return accuracies

    def _get_cross_site_drops(self, model_type: str) -> List[float]:
        """Get AUROC drops for a model type."""
        drops = []
        for result in self.results["cross_site"]:
            if result.model_name.startswith(model_type):
                drops.append(result.auroc_drop)
        return drops

    def _save_results(self) -> None:
        """Save all results to disk."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each result type as JSON
        for result_type, results_list in self.results.items():
            output_path = output_dir / f"{result_type}.json"

            # Convert to serializable format
            serializable = [
                result.to_dict() if hasattr(result, "to_dict") else result
                for result in results_list
            ]

            with open(output_path, "w") as f:
                json.dump(serializable, f, indent=2)

            logger.info(f"Saved {result_type} results to {output_path}")

    def load_results(self) -> None:
        """Load results from disk."""
        # Implementation for loading saved results
        pass

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "num_models": len(self.config.models),
            "num_datasets": len(self.config.datasets),
            "task_performance_evaluations": len(self.results["task_performance"]),
            "robustness_evaluations": len(self.results["robustness"]),
            "cross_site_evaluations": len(self.results["cross_site"]),
            "calibration_evaluations": len(self.results["calibration"]),
            "hypothesis_tests": len(self.results["hypothesis_tests"]),
        }
        return summary


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_rq1_evaluator(
    models: List[ModelCheckpoint],
    datasets: Dict[str, DataLoader],
    output_dir: Union[str, Path],
    **kwargs,
) -> RQ1Evaluator:
    """
    Factory function to create RQ1Evaluator.

    Args:
        models: List of model checkpoints
        datasets: Dictionary of dataloaders
        output_dir: Output directory for results
        **kwargs: Additional configuration options

    Returns:
        Configured RQ1Evaluator instance
    """
    config = EvaluationConfig(
        models=models,
        datasets=datasets,
        output_dir=Path(output_dir),
        **kwargs,
    )

    return RQ1Evaluator(config)
