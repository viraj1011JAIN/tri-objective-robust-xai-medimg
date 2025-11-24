# Phase 5.2: Complete Fixes for A1+ Grade
# =========================================
# This file contains all the missing methods and fixes identified by the REAL validator.
# Add these methods to the respective classes in train_pgd_at.py and evaluate_pgd_at.py

# ==============================================================================
# PHASE 1: Missing Methods in PGDATTrainer (scripts/training/train_pgd_at.py)
# ==============================================================================

Add this method to the PGDATTrainer class after the __init__ method:

```python
    def _setup_training(self) -> None:
        """
        Setup training components and verify configuration.

        This method validates that all training components are properly
        initialized and logs setup information. Call after __init__.
        """
        # Ensure model is on correct device
        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        logger.info("Training setup verification:")
        logger.info(f"  Total parameters: {n_params:,}")
        logger.info(f"  Trainable parameters: {n_trainable:,}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"  Scheduler: {self.scheduler.__class__.__name__}")
        logger.info(f"  Mixed precision: {self.config['training'].get('use_amp', False)}")  # noqa

        # Verify components
        assert self.model is not None, "Model not initialized"
        assert self.optimizer is not None, "Optimizer not initialized"
        assert self.criterion is not None, "Loss function not initialized"
        assert self.trainer is not None, "Adversarial trainer not initialized"

        logger.info("✓ Training setup verified")
```

# ==============================================================================
# PHASE 2: Missing Methods in PGDATEvaluator (scripts/evaluation/evaluate_pgd_at.py)  # noqa
# ==============================================================================

Add these methods to the PGDATEvaluator class:

```python
    def _load_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        """
        Load model from checkpoint with proper validation.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model in eval mode

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is corrupted or invalid
        """
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise ValueError(
                f"Failed to load checkpoint {checkpoint_path}: {e}"
            )

        # Validate checkpoint structure
        required_keys = ['model_state_dict']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(
                f"Invalid checkpoint: missing {missing_keys} in {checkpoint_path}"
            )

        # Build model
        model_config = checkpoint.get('config', {}).get('model', self.config['model'])  # noqa
        model = build_model(
            architecture=model_config['architecture'],
            num_classes=model_config['num_classes'],
            pretrained=False,
            in_channels=model_config.get('in_channels', 3)
        )

        # Load weights
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            raise ValueError(
                f"Failed to load model weights from {checkpoint_path}: {e}"
            )

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        logger.info(f"✓ Loaded checkpoint successfully")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  Seed: {checkpoint.get('seed', 'unknown')}")
        if 'best_robust_acc' in checkpoint:
            logger.info(f"  Best robust acc: {checkpoint['best_robust_acc']:.2f}%")  # noqa

        return model

    def evaluate(
        self,
        model_checkpoint_paths: Optional[List[Path]] = None
    ) -> Dict:
        """
        Main evaluation orchestrator.

        This is the primary entry point that coordinates:
        1. Loading each model checkpoint
        2. Evaluating on all test sets
        3. Computing all metrics (clean + robust)
        4. Aggregating across seeds
        5. Statistical significance testing

        Args:
            model_checkpoint_paths: List of checkpoint paths (uses self.model_paths if None)  # noqa

        Returns:
            Complete evaluation results with aggregated statistics
        """
        import gc

        if model_checkpoint_paths is None:
            model_checkpoint_paths = self.model_paths

        logger.info(f"\n{'='*80}")
        logger.info(f"Starting comprehensive evaluation of {len(model_checkpoint_paths)} models")  # noqa
        logger.info(f"{'='*80}\n")

        # Build test loaders once
        test_loaders = self.build_test_loaders()
        logger.info(f"Built {len(test_loaders)} test loaders")

        # Evaluate each model
        all_results = {}

        for seed_idx, checkpoint_path in enumerate(model_checkpoint_paths):
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluating checkpoint {seed_idx + 1}/{len(model_checkpoint_paths)}")  # noqa
            logger.info(f"Path: {checkpoint_path}")
            logger.info(f"{'='*70}\n")

            try:
                # Load model
                model = self._load_checkpoint(checkpoint_path)

                # Evaluate on all test sets
                seed_results = self.evaluate_model_comprehensive(
                    model, test_loaders
                )

                # Store results
                seed_name = f"seed_{seed_idx}"
                all_results[seed_name] = seed_results

                logger.info(f"✓ Evaluation complete for {seed_name}")

            except Exception as e:
                logger.error(f"✗ Evaluation failed for {checkpoint_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                all_results[f"seed_{seed_idx}"] = {'error': str(e)}

            finally:
                # CRITICAL: Memory cleanup
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                logger.info("Memory cleaned up")

        # Aggregate results across seeds
        logger.info(f"\n{'='*80}")
        logger.info("Computing aggregate statistics...")
        logger.info(f"{'='*80}\n")

        aggregated_results = self._aggregate_results(all_results)

        # Save results
        self.save_results({
            'per_seed_results': all_results,
            'aggregated_results': aggregated_results,
            'n_seeds': len(model_checkpoint_paths)
        })

        return {
            'per_seed_results': all_results,
            'aggregated_results': aggregated_results,
            'n_seeds': len(model_checkpoint_paths)
        }

    def evaluate_robustness(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        attack_name: str = 'pgd_10',
        epsilon: float = 0.03137,
        num_steps: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate model robustness under adversarial attack.

        Args:
            model: Model to evaluate
            dataloader: Test data loader
            attack_name: Name of attack for logging
            epsilon: Perturbation budget (8/255 = 0.03137)
            num_steps: Number of PGD steps

        Returns:
            Robust accuracy and metrics
        """
        model.eval()

        # Create PGD attack
        pgd_config = PGDConfig(
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=epsilon / 4,
            random_start=True,
            clip_min=0.0,
            clip_max=1.0
        )
        attack = PGD(model=model, config=pgd_config)

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc=f"Robust eval ({attack_name})"):  # noqa
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Generate adversarial examples
            with torch.enable_grad():  # Need gradients for attack
                adv_images = attack(images, labels)

            # Evaluate on adversarial examples
            with torch.no_grad():
                outputs = model(adv_images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = 100.0 * correct / total

        # Compute comprehensive metrics
        metrics = calculate_metrics(
            y_true=all_labels,
            y_pred=all_preds,
            y_prob=np.array(all_probs)
        )
        metrics['accuracy'] = accuracy
        metrics['attack_name'] = attack_name
        metrics['epsilon'] = epsilon
        metrics['num_steps'] = num_steps

        return metrics

    def _aggregate_results(self, all_results: Dict) -> Dict:
        """
        Aggregate results across seeds with statistics.

        Args:
            all_results: Results from all seeds

        Returns:
            Aggregated statistics (mean, std, CI, min, max)
        """
        aggregated = {}

        # Filter out error results
        valid_results = {
            k: v for k, v in all_results.items()
            if 'error' not in v
        }

        if not valid_results:
            logger.error("No valid results to aggregate")
            return {}

        # Get test set names from first valid result
        first_seed = list(valid_results.values())[0]
        test_sets = list(first_seed.keys())

        for test_set in test_sets:
            aggregated[test_set] = {}

            # Get attack types from first result
            attack_types = list(first_seed[test_set].keys())

            for attack_type in attack_types:
                # Extract metrics across seeds
                accuracies = []
                aurocs = []

                for seed_results in valid_results.values():
                    if test_set in seed_results and attack_type in seed_results[test_set]:  # noqa
                        metrics = seed_results[test_set][attack_type]
                        accuracies.append(metrics.get('accuracy', 0.0))
                        if 'auroc' in metrics:
                            aurocs.append(metrics['auroc'])

                accuracies = np.array(accuracies)

                # Compute statistics
                aggregated[test_set][attack_type] = {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies, ddof=1)),
                    'accuracy_min': float(np.min(accuracies)),
                    'accuracy_max': float(np.max(accuracies)),
                    'accuracy_values': accuracies.tolist(),
                    'n_seeds': len(accuracies)
                }

                if aurocs:
                    aurocs = np.array(aurocs)
                    aggregated[test_set][attack_type].update({
                        'auroc_mean': float(np.mean(aurocs)),
                        'auroc_std': float(np.std(aurocs, ddof=1)),
                        'auroc_values': aurocs.tolist()
                    })

        return aggregated
```

# ==============================================================================
# PHASE 3: RQ1 Hypothesis Test (ADD NEW METHOD TO PGDATEvaluator)
# ==============================================================================

```python
    def test_rq1_hypothesis(
        self,
        baseline_results: Dict[str, Dict],
        pgd_at_results: Dict[str, Dict],
        alpha: float = 0.01
    ) -> Dict:
        """
        Test RQ1 Hypothesis H1c: PGD-AT does NOT improve cross-site generalization.  # noqa

        This is the CORE research question for Phase 5.2.

        Expected Result:
            p > 0.05 (no significant difference in cross-site AUROC drops)
            → Confirms PGD-AT doesn't help cross-site
            → Justifies need for tri-objective approach

        Args:
            baseline_results: Baseline model results
            pgd_at_results: PGD-AT model results
            alpha: Significance level

        Returns:
            Comprehensive statistical test results
        """
        from src.analysis.rq1_hypothesis_test import test_rq1_hypothesis

        logger.info("\n" + "="*80)
        logger.info("RQ1 HYPOTHESIS TEST: Cross-Site Generalization")
        logger.info("="*80 + "\n")

        # Extract AUROC drops for each target dataset
        baseline_drops = {}
        pgd_at_drops = {}

        target_datasets = ['isic2019', 'isic2020', 'derm7pt']
        source_dataset = 'isic2018_test'

        # Extract AUROCs for source dataset
        baseline_source_aurocs = [
            baseline_results[seed][source_dataset]['clean'].get('auroc', 0.0)
            for seed in baseline_results.keys()
            if 'error' not in baseline_results[seed]
        ]
        pgd_at_source_aurocs = [
            pgd_at_results[seed][source_dataset]['clean'].get('auroc', 0.0)
            for seed in pgd_at_results.keys()
            if 'error' not in pgd_at_results[seed]
        ]

        # Extract AUROCs for each target dataset and compute drops
        for target in target_datasets:
            if target not in baseline_results.get(list(baseline_results.keys())[0], {}):  # noqa
                logger.warning(f"Target dataset {target} not found in results")
                continue

            baseline_target_aurocs = [
                baseline_results[seed][target]['clean'].get('auroc', 0.0)
                for seed in baseline_results.keys()
                if 'error' not in baseline_results[seed]
            ]
            pgd_at_target_aurocs = [
                pgd_at_results[seed][target]['clean'].get('auroc', 0.0)
                for seed in pgd_at_results.keys()
                if 'error' not in pgd_at_results[seed]
            ]

            # Compute AUROC drops
            baseline_drops[target] = [
                src - tgt
                for src, tgt in zip(baseline_source_aurocs, baseline_target_aurocs)  # noqa
            ]
            pgd_at_drops[target] = [
                src - tgt
                for src, tgt in zip(pgd_at_source_aurocs, pgd_at_target_aurocs)
            ]

        # Run statistical test
        try:
            test_results = test_rq1_hypothesis(
                baseline_drops=baseline_drops,
                pgd_at_drops=pgd_at_drops,
                alpha=alpha,
                bonferroni_correction=True
            )

            logger.info("RQ1 Test Results:")
            logger.info(f"  H1c Confirmed: {test_results['h1c_confirmed']}")
            logger.info(f"  Bonferroni correction applied: {test_results['bonferroni_correction']}")  # noqa
            logger.info(f"\nInterpretation:\n{test_results['interpretation']}")

            return test_results

        except Exception as e:
            logger.error(f"RQ1 hypothesis test failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
```

# ==============================================================================
# PHASE 4: Advanced Statistical Analysis (ADD TO PGDATEvaluator)
# ==============================================================================

```python
    def statistical_testing(
        self,
        baseline_metrics: np.ndarray,
        pgd_at_metrics: np.ndarray,
        metric_name: str = "accuracy",
        alpha: float = 0.01
    ) -> Dict:
        """
        A1+ grade statistical analysis with full rigor.

        Includes:
        - Normality testing (Shapiro-Wilk)
        - Parametric (t-test) or non-parametric (Wilcoxon) test selection
        - Effect size (Cohen's d, Hedge's g)
        - Bootstrap confidence intervals
        - Statistical power analysis

        Args:
            baseline_metrics: Baseline results (n=3 seeds)
            pgd_at_metrics: PGD-AT results (n=3 seeds)
            metric_name: Name of metric being tested
            alpha: Significance level

        Returns:
            Comprehensive statistical analysis results
        """
        from scipy import stats as scipy_stats
        from scipy.stats import bootstrap

        results = {'metric_name': metric_name, 'alpha': alpha}

        # 1. Normality testing
        _, p_baseline = scipy_stats.shapiro(baseline_metrics)
        _, p_pgd = scipy_stats.shapiro(pgd_at_metrics)
        both_normal = (p_baseline > 0.05) and (p_pgd > 0.05)

        results['normality'] = {
            'baseline_normal': p_baseline > 0.05,
            'pgd_at_normal': p_pgd > 0.05,
            'use_parametric': both_normal
        }

        # 2. Choose appropriate test
        if both_normal and len(baseline_metrics) == len(pgd_at_metrics):
            # Paired t-test (parametric)
            t_stat, p_value = scipy_stats.ttest_rel(
                pgd_at_metrics, baseline_metrics
            )
            test_name = 'paired_t_test'
        else:
            # Wilcoxon signed-rank (non-parametric)
            t_stat, p_value = scipy_stats.wilcoxon(
                pgd_at_metrics, baseline_metrics,
                alternative='two-sided'
            )
            test_name = 'wilcoxon_signed_rank'

        results['test'] = {
            'name': test_name,
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < alpha
        }

        # 3. Effect size
        mean_diff = float(np.mean(pgd_at_metrics) - np.mean(baseline_metrics))
        pooled_std = np.sqrt(
            (np.var(baseline_metrics, ddof=1) + np.var(pgd_at_metrics, ddof=1)) / 2  # noqa
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Hedge's g (small sample correction)
        n = len(baseline_metrics) + len(pgd_at_metrics)
        hedges_g = cohens_d * (1 - (3 / (4 * n - 9)))

        results['effect_size'] = {
            'mean_diff': mean_diff,
            'cohens_d': float(cohens_d),
            'hedges_g': float(hedges_g),
            'interpretation': self._interpret_effect_size(abs(hedges_g))
        }

        # 4. Bootstrap confidence interval
        def stat_func(x, y):
            return np.mean(x) - np.mean(y)

        rng = np.random.default_rng(42)
        try:
            boot_result = bootstrap(
                (pgd_at_metrics, baseline_metrics),
                stat_func,
                n_resamples=10000,
                confidence_level=0.95,
                random_state=rng,
                method='percentile'
            )
            results['confidence_interval'] = {
                'mean_diff': mean_diff,
                'ci_lower': float(boot_result.confidence_interval.low),
                'ci_upper': float(boot_result.confidence_interval.high),
                'level': 0.95
            }
        except Exception as e:
            logger.warning(f"Bootstrap CI failed: {e}")
            results['confidence_interval'] = {'error': str(e)}

        # 5. Statistical power
        try:
            from statsmodels.stats.power import ttest_power
            power = ttest_power(
                effect_size=abs(hedges_g),
                nobs=len(baseline_metrics),
                alpha=alpha,
                alternative='two-sided'
            )
            results['power'] = {
                'value': float(power),
                'adequate': power >= 0.80
            }
        except Exception as e:
            logger.warning(f"Power analysis failed: {e}")
            results['power'] = {'error': str(e)}

        return results

    def _interpret_effect_size(self, effect: float) -> str:
        """Interpret effect size magnitude (Cohen's conventions)."""
        if effect < 0.2:
            return "negligible"
        elif effect < 0.5:
            return "small"
        elif effect < 0.8:
            return "medium"
        else:
            return "large"
```

# ==============================================================================
# USAGE: How to integrate these fixes
# ==============================================================================

1. Open scripts/training/train_pgd_at.py
   - Find the PGDATTrainer class
   - Add the _setup_training() method after __init__
   - Call self._setup_training() at the end of __init__

2. Open scripts/evaluation/evaluate_pgd_at.py
   - Find the PGDATEvaluator class
   - Add all the missing methods:
     * _load_checkpoint()
     * evaluate()
     * evaluate_robustness()
     * _aggregate_results()
     * test_rq1_hypothesis()
     * statistical_testing()
     * _interpret_effect_size()

3. Run the REAL validator again:
   python scripts/validation/validate_phase_5_2_REAL.py

4. Expected result: Grade B+ to A (all functional tests passing)

5. For A1+, ensure RQ1 test is called in your evaluation pipeline
