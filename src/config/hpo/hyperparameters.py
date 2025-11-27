"""
Hyperparameter Configuration System for Tri-Objective Robust XAI Medical Imaging.

This module provides a comprehensive dataclass-based configuration system for managing
hyperparameters across different model architectures, training strategies, and optimization
objectives. It includes validation, serialization, and factory methods for creating
configurations.

Author: Viraj Jain
Date: November 2025
"""

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


class ActivationType(str, Enum):
    """Supported activation functions."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"
    SELU = "selu"
    SWISH = "swish"
    MISH = "mish"
    TANH = "tanh"
    SIGMOID = "sigmoid"


class OptimizerType(str, Enum):
    """Supported optimizer types."""

    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAMAX = "adamax"
    NADAM = "nadam"


class SchedulerType(str, Enum):
    """Supported learning rate schedulers."""

    STEP = "step"
    MULTISTEP = "multistep"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    CYCLIC = "cyclic"
    ONE_CYCLE = "one_cycle"


class NormalizationType(str, Enum):
    """Supported normalization layers."""

    BATCH_NORM = "batch_norm"
    INSTANCE_NORM = "instance_norm"
    LAYER_NORM = "layer_norm"
    GROUP_NORM = "group_norm"
    NONE = "none"


class AttackType(str, Enum):
    """Supported adversarial attack types."""

    FGSM = "fgsm"
    PGD = "pgd"
    CW = "cw"
    DEEPFOOL = "deepfool"
    AUTO_ATTACK = "auto_attack"


@dataclass
class ModelHyperparameters:
    """
    Model architecture hyperparameters.

    Attributes:
        architecture: Model architecture name (resnet50, densenet121, efficientnet_b0, etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout_rate: Dropout probability
        activation: Activation function type
        normalization: Normalization layer type
        use_attention: Whether to use attention mechanisms
        attention_heads: Number of attention heads if use_attention is True
        hidden_dims: List of hidden layer dimensions
        use_residual: Whether to use residual connections
        use_se_blocks: Whether to use Squeeze-and-Excitation blocks
        se_reduction: SE block reduction ratio
    """

    architecture: str = "resnet50"
    num_classes: int = 2
    pretrained: bool = True
    dropout_rate: float = 0.5
    activation: ActivationType = ActivationType.RELU
    normalization: NormalizationType = NormalizationType.BATCH_NORM
    use_attention: bool = False
    attention_heads: int = 8
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    use_residual: bool = True
    use_se_blocks: bool = False
    se_reduction: int = 16

    def __post_init__(self):
        """Validate hyperparameters after initialization."""
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError(f"dropout_rate must be in [0, 1], got {self.dropout_rate}")
        if self.attention_heads < 1:
            raise ValueError(
                f"attention_heads must be >= 1, got {self.attention_heads}"
            )
        if self.se_reduction < 1:
            raise ValueError(f"se_reduction must be >= 1, got {self.se_reduction}")
        if not all(dim > 0 for dim in self.hidden_dims):
            raise ValueError("All hidden_dims must be positive")

        # Convert string enums if needed
        if isinstance(self.activation, str):
            self.activation = ActivationType(self.activation)
        if isinstance(self.normalization, str):
            self.normalization = NormalizationType(self.normalization)


@dataclass
class OptimizerHyperparameters:
    """
    Optimizer hyperparameters.

    Attributes:
        optimizer_type: Type of optimizer
        learning_rate: Initial learning rate
        weight_decay: L2 regularization coefficient
        momentum: Momentum factor (for SGD, RMSprop)
        betas: Coefficients for computing running averages (for Adam variants)
        eps: Term added to denominator for numerical stability
        amsgrad: Whether to use AMSGrad variant (for Adam variants)
        dampening: Dampening for momentum (for SGD)
        nesterov: Whether to use Nesterov momentum (for SGD)
        centered: Whether to use centered RMSprop
    """

    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    dampening: float = 0.0
    nesterov: bool = False
    centered: bool = False

    def __post_init__(self):
        """Validate optimizer hyperparameters."""
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {self.weight_decay}"
            )
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")
        if len(self.betas) != 2:
            raise ValueError(f"betas must have length 2, got {len(self.betas)}")
        if not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"betas must be in [0, 1), got {self.betas}")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}")

        # Convert string enum if needed
        if isinstance(self.optimizer_type, str):
            self.optimizer_type = OptimizerType(self.optimizer_type)


@dataclass
class SchedulerHyperparameters:
    """
    Learning rate scheduler hyperparameters.

    Attributes:
        scheduler_type: Type of scheduler
        step_size: Period of learning rate decay (for StepLR)
        gamma: Multiplicative factor of learning rate decay
        milestones: List of epoch indices for MultiStepLR
        T_max: Maximum number of iterations for CosineAnnealingLR
        T_0: Number of iterations for first restart (CosineAnnealingWarmRestarts)
        T_mult: Factor to increase T_i after restart
        eta_min: Minimum learning rate
        patience: Number of epochs with no improvement (ReduceLROnPlateau)
        factor: Factor by which to reduce learning rate (ReduceLROnPlateau)
        threshold: Threshold for measuring new optimum (ReduceLROnPlateau)
        base_lr: Initial learning rate for cyclic schedulers
        max_lr: Upper learning rate boundary for cyclic schedulers
        step_size_up: Number of iterations in increasing half of cycle
        step_size_down: Number of iterations in decreasing half of cycle
        mode: One of {triangular, triangular2, exp_range} for CyclicLR
        warmup_epochs: Number of warmup epochs
    """

    scheduler_type: SchedulerType = SchedulerType.COSINE
    step_size: int = 30
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [60, 120, 160])
    T_max: int = 200
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-6
    patience: int = 10
    factor: float = 0.1
    threshold: float = 1e-4
    base_lr: float = 1e-5
    max_lr: float = 1e-2
    step_size_up: int = 2000
    step_size_down: Optional[int] = None
    mode: str = "triangular"
    warmup_epochs: int = 5

    def __post_init__(self):
        """Validate scheduler hyperparameters."""
        if self.step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {self.step_size}")
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")
        if self.T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {self.T_max}")
        if self.T_0 < 1:
            raise ValueError(f"T_0 must be >= 1, got {self.T_0}")
        if self.T_mult < 1:
            raise ValueError(f"T_mult must be >= 1, got {self.T_mult}")
        if self.eta_min < 0:
            raise ValueError(f"eta_min must be non-negative, got {self.eta_min}")
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.factor <= 0 or self.factor >= 1:
            raise ValueError(f"factor must be in (0, 1), got {self.factor}")
        if self.warmup_epochs < 0:
            raise ValueError(
                f"warmup_epochs must be non-negative, got {self.warmup_epochs}"
            )

        # Convert string enum if needed
        if isinstance(self.scheduler_type, str):
            self.scheduler_type = SchedulerType(self.scheduler_type)


@dataclass
class TrainingHyperparameters:
    """
    Training process hyperparameters.

    Attributes:
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        gradient_clip_value: Maximum gradient norm (0 means no clipping)
        mixed_precision: Whether to use mixed precision training
        accumulation_steps: Number of gradient accumulation steps
        early_stopping_patience: Patience for early stopping (0 means no early stopping)
        early_stopping_delta: Minimum change to qualify as improvement
        val_frequency: Validation frequency in epochs
        log_frequency: Logging frequency in steps
        save_frequency: Checkpoint save frequency in epochs
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory in data loader
        seed: Random seed for reproducibility
    """

    batch_size: int = 32
    num_epochs: int = 200
    gradient_clip_value: float = 1.0
    mixed_precision: bool = True
    accumulation_steps: int = 1
    early_stopping_patience: int = 20
    early_stopping_delta: float = 1e-4
    val_frequency: int = 1
    log_frequency: int = 100
    save_frequency: int = 10
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42

    def __post_init__(self):
        """Validate training hyperparameters."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if self.gradient_clip_value < 0:
            raise ValueError(
                f"gradient_clip_value must be non-negative, got {self.gradient_clip_value}"
            )
        if self.accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {self.accumulation_steps}"
            )
        if self.early_stopping_patience < 0:
            raise ValueError(
                f"early_stopping_patience must be non-negative, got {self.early_stopping_patience}"
            )
        if self.val_frequency < 1:
            raise ValueError(f"val_frequency must be >= 1, got {self.val_frequency}")
        if self.log_frequency < 1:
            raise ValueError(f"log_frequency must be >= 1, got {self.log_frequency}")
        if self.save_frequency < 1:
            raise ValueError(f"save_frequency must be >= 1, got {self.save_frequency}")
        if self.num_workers < 0:
            raise ValueError(
                f"num_workers must be non-negative, got {self.num_workers}"
            )


@dataclass
class RobustnessHyperparameters:
    """
    Adversarial robustness hyperparameters.

    Attributes:
        enable_adversarial_training: Whether to use adversarial training
        attack_type: Type of adversarial attack
        epsilon: Perturbation budget
        alpha: Step size for iterative attacks
        num_steps: Number of attack steps
        random_start: Whether to use random initialization
        targeted: Whether to use targeted attacks
        adversarial_ratio: Ratio of adversarial examples in batch
        trades_beta: Beta parameter for TRADES loss
        mart_beta: Beta parameter for MART loss
        use_trades: Whether to use TRADES objective
        use_mart: Whether to use MART objective
    """

    enable_adversarial_training: bool = True
    attack_type: AttackType = AttackType.PGD
    epsilon: float = 8.0 / 255.0
    alpha: float = 2.0 / 255.0
    num_steps: int = 10
    random_start: bool = True
    targeted: bool = False
    adversarial_ratio: float = 1.0
    trades_beta: float = 6.0
    mart_beta: float = 6.0
    use_trades: bool = False
    use_mart: bool = False

    def __post_init__(self):
        """Validate robustness hyperparameters."""
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {self.epsilon}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if not 0.0 <= self.adversarial_ratio <= 1.0:
            raise ValueError(
                f"adversarial_ratio must be in [0, 1], got {self.adversarial_ratio}"
            )
        if self.trades_beta < 0:
            raise ValueError(
                f"trades_beta must be non-negative, got {self.trades_beta}"
            )
        if self.mart_beta < 0:
            raise ValueError(f"mart_beta must be non-negative, got {self.mart_beta}")

        # Convert string enum if needed
        if isinstance(self.attack_type, str):
            self.attack_type = AttackType(self.attack_type)


@dataclass
class ExplainabilityHyperparameters:
    """
    Explainability hyperparameters.

    Attributes:
        enable_xai_loss: Whether to include XAI loss in training
        xai_loss_weight: Weight for XAI loss term
        use_gradcam: Whether to use GradCAM
        use_integrated_gradients: Whether to use Integrated Gradients
        use_lime: Whether to use LIME
        use_shap: Whether to use SHAP
        ig_steps: Number of steps for Integrated Gradients
        lime_num_samples: Number of samples for LIME
        shap_num_samples: Number of samples for SHAP
        concept_coherence_weight: Weight for concept coherence loss
        spatial_coherence_weight: Weight for spatial coherence loss
        faithfulness_weight: Weight for faithfulness loss
        sparsity_weight: Weight for sparsity regularization
    """

    enable_xai_loss: bool = True
    xai_loss_weight: float = 0.1
    use_gradcam: bool = True
    use_integrated_gradients: bool = True
    use_lime: bool = False
    use_shap: bool = False
    ig_steps: int = 50
    lime_num_samples: int = 1000
    shap_num_samples: int = 100
    concept_coherence_weight: float = 0.05
    spatial_coherence_weight: float = 0.05
    faithfulness_weight: float = 0.1
    sparsity_weight: float = 0.01

    def __post_init__(self):
        """Validate explainability hyperparameters."""
        if self.xai_loss_weight < 0:
            raise ValueError(
                f"xai_loss_weight must be non-negative, got {self.xai_loss_weight}"
            )
        if self.ig_steps < 1:
            raise ValueError(f"ig_steps must be >= 1, got {self.ig_steps}")
        if self.lime_num_samples < 1:
            raise ValueError(
                f"lime_num_samples must be >= 1, got {self.lime_num_samples}"
            )
        if self.shap_num_samples < 1:
            raise ValueError(
                f"shap_num_samples must be >= 1, got {self.shap_num_samples}"
            )
        if self.concept_coherence_weight < 0:
            raise ValueError(f"concept_coherence_weight must be non-negative")
        if self.spatial_coherence_weight < 0:
            raise ValueError(f"spatial_coherence_weight must be non-negative")
        if self.faithfulness_weight < 0:
            raise ValueError(f"faithfulness_weight must be non-negative")
        if self.sparsity_weight < 0:
            raise ValueError(f"sparsity_weight must be non-negative")


@dataclass
class TriObjectiveHyperparameters:
    """
    Tri-objective optimization hyperparameters.

    Attributes:
        accuracy_weight: Weight for accuracy objective
        robustness_weight: Weight for robustness objective
        explainability_weight: Weight for explainability objective
        use_dynamic_weighting: Whether to use dynamic weight adjustment
        weight_adjustment_frequency: Frequency of weight adjustment in epochs
        min_weight: Minimum weight value
        max_weight: Maximum weight value
        pareto_alpha: Alpha parameter for Pareto optimization
    """

    accuracy_weight: float = 1.0
    robustness_weight: float = 1.0
    explainability_weight: float = 1.0
    use_dynamic_weighting: bool = False
    weight_adjustment_frequency: int = 10
    min_weight: float = 0.1
    max_weight: float = 2.0
    pareto_alpha: float = 0.5

    def __post_init__(self):
        """Validate tri-objective hyperparameters."""
        if self.accuracy_weight < 0:
            raise ValueError(
                f"accuracy_weight must be non-negative, got {self.accuracy_weight}"
            )
        if self.robustness_weight < 0:
            raise ValueError(
                f"robustness_weight must be non-negative, got {self.robustness_weight}"
            )
        if self.explainability_weight < 0:
            raise ValueError(
                f"explainability_weight must be non-negative, got {self.explainability_weight}"
            )
        if self.weight_adjustment_frequency < 1:
            raise ValueError(f"weight_adjustment_frequency must be >= 1")
        if self.min_weight <= 0:
            raise ValueError(f"min_weight must be positive, got {self.min_weight}")
        if self.max_weight <= self.min_weight:
            raise ValueError(f"max_weight must be > min_weight")
        if not 0.0 <= self.pareto_alpha <= 1.0:
            raise ValueError(f"pareto_alpha must be in [0, 1], got {self.pareto_alpha}")


@dataclass
class HyperparameterConfig:
    """
    Complete hyperparameter configuration.

    This is the main configuration class that aggregates all hyperparameter categories.

    Attributes:
        model: Model architecture hyperparameters
        optimizer: Optimizer hyperparameters
        scheduler: Learning rate scheduler hyperparameters
        training: Training process hyperparameters
        robustness: Adversarial robustness hyperparameters
        explainability: Explainability hyperparameters
        tri_objective: Tri-objective optimization hyperparameters
        experiment_name: Name of the experiment
        description: Description of the configuration
    """

    model: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    optimizer: OptimizerHyperparameters = field(
        default_factory=OptimizerHyperparameters
    )
    scheduler: SchedulerHyperparameters = field(
        default_factory=SchedulerHyperparameters
    )
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    robustness: RobustnessHyperparameters = field(
        default_factory=RobustnessHyperparameters
    )
    explainability: ExplainabilityHyperparameters = field(
        default_factory=ExplainabilityHyperparameters
    )
    tri_objective: TriObjectiveHyperparameters = field(
        default_factory=TriObjectiveHyperparameters
    )
    experiment_name: str = "default_experiment"
    description: str = "Default hyperparameter configuration"

    def __post_init__(self):
        """Validate complete configuration."""
        # Ensure all components are properly instantiated
        if not isinstance(self.model, ModelHyperparameters):
            self.model = ModelHyperparameters(
                **self.model if isinstance(self.model, dict) else {}
            )
        if not isinstance(self.optimizer, OptimizerHyperparameters):
            self.optimizer = OptimizerHyperparameters(
                **self.optimizer if isinstance(self.optimizer, dict) else {}
            )
        if not isinstance(self.scheduler, SchedulerHyperparameters):
            self.scheduler = SchedulerHyperparameters(
                **self.scheduler if isinstance(self.scheduler, dict) else {}
            )
        if not isinstance(self.training, TrainingHyperparameters):
            self.training = TrainingHyperparameters(
                **self.training if isinstance(self.training, dict) else {}
            )
        if not isinstance(self.robustness, RobustnessHyperparameters):
            self.robustness = RobustnessHyperparameters(
                **self.robustness if isinstance(self.robustness, dict) else {}
            )
        if not isinstance(self.explainability, ExplainabilityHyperparameters):
            self.explainability = ExplainabilityHyperparameters(
                **self.explainability if isinstance(self.explainability, dict) else {}
            )
        if not isinstance(self.tri_objective, TriObjectiveHyperparameters):
            self.tri_objective = TriObjectiveHyperparameters(
                **self.tri_objective if isinstance(self.tri_objective, dict) else {}
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """

        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        return convert_enums(asdict(self))

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Path to save the YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save the JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HyperparameterConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            HyperparameterConfig instance
        """
        return cls(
            model=ModelHyperparameters(**config_dict.get("model", {})),
            optimizer=OptimizerHyperparameters(**config_dict.get("optimizer", {})),
            scheduler=SchedulerHyperparameters(**config_dict.get("scheduler", {})),
            training=TrainingHyperparameters(**config_dict.get("training", {})),
            robustness=RobustnessHyperparameters(**config_dict.get("robustness", {})),
            explainability=ExplainabilityHyperparameters(
                **config_dict.get("explainability", {})
            ),
            tri_objective=TriObjectiveHyperparameters(
                **config_dict.get("tri_objective", {})
            ),
            experiment_name=config_dict.get("experiment_name", "default_experiment"),
            description=config_dict.get(
                "description", "Default hyperparameter configuration"
            ),
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "HyperparameterConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            HyperparameterConfig instance
        """
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HyperparameterConfig":
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            HyperparameterConfig instance
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def create_baseline(cls) -> "HyperparameterConfig":
        """
        Create baseline configuration for standard training.

        Returns:
            HyperparameterConfig with baseline settings
        """
        return cls(
            experiment_name="baseline_config",
            description="Baseline configuration for standard training",
        )

    @classmethod
    def create_high_accuracy(cls) -> "HyperparameterConfig":
        """
        Create configuration optimized for high accuracy.

        Returns:
            HyperparameterConfig with accuracy-focused settings
        """
        return cls(
            model=ModelHyperparameters(
                architecture="efficientnet_b4", dropout_rate=0.3, use_se_blocks=True
            ),
            optimizer=OptimizerHyperparameters(
                optimizer_type=OptimizerType.ADAMW,
                learning_rate=0.0001,
                weight_decay=0.05,
            ),
            training=TrainingHyperparameters(
                batch_size=16,
                num_epochs=300,
                mixed_precision=True,
                early_stopping_patience=30,
            ),
            tri_objective=TriObjectiveHyperparameters(
                accuracy_weight=2.0, robustness_weight=0.5, explainability_weight=0.5
            ),
            experiment_name="high_accuracy_config",
            description="Configuration optimized for maximum accuracy",
        )

    @classmethod
    def create_high_robustness(cls) -> "HyperparameterConfig":
        """
        Create configuration optimized for adversarial robustness.

        Returns:
            HyperparameterConfig with robustness-focused settings
        """
        return cls(
            model=ModelHyperparameters(
                architecture="resnet50", dropout_rate=0.2, use_residual=True
            ),
            robustness=RobustnessHyperparameters(
                enable_adversarial_training=True,
                attack_type=AttackType.PGD,
                epsilon=8.0 / 255.0,
                num_steps=20,
                use_trades=True,
                trades_beta=8.0,
            ),
            training=TrainingHyperparameters(batch_size=64, num_epochs=200),
            tri_objective=TriObjectiveHyperparameters(
                accuracy_weight=0.5, robustness_weight=2.0, explainability_weight=0.5
            ),
            experiment_name="high_robustness_config",
            description="Configuration optimized for adversarial robustness",
        )

    @classmethod
    def create_high_explainability(cls) -> "HyperparameterConfig":
        """
        Create configuration optimized for explainability.

        Returns:
            HyperparameterConfig with explainability-focused settings
        """
        return cls(
            model=ModelHyperparameters(
                architecture="resnet34", use_attention=True, attention_heads=8
            ),
            explainability=ExplainabilityHyperparameters(
                enable_xai_loss=True,
                xai_loss_weight=0.2,
                use_gradcam=True,
                use_integrated_gradients=True,
                concept_coherence_weight=0.1,
                spatial_coherence_weight=0.1,
                faithfulness_weight=0.15,
            ),
            tri_objective=TriObjectiveHyperparameters(
                accuracy_weight=0.5, robustness_weight=0.5, explainability_weight=2.0
            ),
            experiment_name="high_explainability_config",
            description="Configuration optimized for explainability",
        )

    @classmethod
    def create_balanced(cls) -> "HyperparameterConfig":
        """
        Create balanced configuration for tri-objective optimization.

        Returns:
            HyperparameterConfig with balanced settings
        """
        return cls(
            model=ModelHyperparameters(
                architecture="resnet50",
                dropout_rate=0.4,
                use_attention=True,
                use_se_blocks=True,
            ),
            robustness=RobustnessHyperparameters(
                enable_adversarial_training=True, epsilon=8.0 / 255.0, use_trades=True
            ),
            explainability=ExplainabilityHyperparameters(
                enable_xai_loss=True, xai_loss_weight=0.1, use_gradcam=True
            ),
            tri_objective=TriObjectiveHyperparameters(
                accuracy_weight=1.0,
                robustness_weight=1.0,
                explainability_weight=1.0,
                use_dynamic_weighting=True,
            ),
            experiment_name="balanced_config",
            description="Balanced configuration for tri-objective optimization",
        )


def validate_config(config: HyperparameterConfig) -> Tuple[bool, List[str]]:
    """
    Validate a hyperparameter configuration.

    Args:
        config: Configuration to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        # Try to convert to dict and back
        config_dict = config.to_dict()
        _ = HyperparameterConfig.from_dict(config_dict)
    except Exception as e:
        errors.append(f"Configuration serialization error: {str(e)}")

    # Check for logical inconsistencies
    if config.robustness.enable_adversarial_training and config.robustness.epsilon == 0:
        errors.append("Adversarial training enabled but epsilon is 0")

    if (
        config.explainability.enable_xai_loss
        and config.explainability.xai_loss_weight == 0
    ):
        errors.append("XAI loss enabled but weight is 0")

    if config.tri_objective.use_dynamic_weighting:
        if (
            config.tri_objective.weight_adjustment_frequency
            > config.training.num_epochs
        ):
            errors.append("Weight adjustment frequency exceeds total epochs")

    return len(errors) == 0, errors
