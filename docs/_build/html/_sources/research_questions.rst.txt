Research Questions and Experimental Design
==========================================

This project is structured around three main research questions (RQ1–RQ3).

RQ1: Robustness and Cross-Site Generalisation
---------------------------------------------

Can adversarial robustness and cross-site generalisation be jointly optimised
through a unified training objective?

High-level components:

- Baselines: standard training, PGD adversarial training, TRADES.
- Tri-objective models: task + robustness + explanation stability.
- Metrics: clean accuracy, robust accuracy under PGD, cross-site AUROC drop.
- Analysis: Pareto frontiers and statistical tests.

RQ2: Concept-Grounded Explanation Stability
-------------------------------------------

Does concept regularisation produce explanations that are both adversarially
stable and clinically meaningful?

High-level components:

- Explanation method: Grad-CAM (and possible extensions).
- Concept grounding: TCAV on medical and artifact concepts.
- Metrics: SSIM between clean and adversarial heatmaps, TCAV scores.
- Analysis: visual comparison, stability under attack, statistical tests.

RQ3: Safe Selective Prediction
------------------------------

Can multi-signal gating enable safe selective prediction for medical imaging
models?

High-level components:

- Signals: predictive confidence, robustness indicators, explanation stability.
- Mechanism: selective prediction with reject option.
- Metrics: coverage-accuracy curves, risk on accepted vs. rejected cases.
- Analysis: impact on cross-site performance and calibration.

These research questions guide the configuration files under
``configs/experiments/`` and the evaluation scripts under ``scripts/``.
