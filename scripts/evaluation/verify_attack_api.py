"""
Verification Script: Attack API and Configuration
==================================================

This script verifies that all attack implementations exist and have the
correct API before running the full evaluation.

Tests:
1. Import all attack classes
2. Verify .generate() method exists
3. Test PGD with step_size parameter
4. Verify device handling
5. Test basic attack execution

Run this BEFORE running evaluate_baseline_robustness.py

Author: Viraj Pankaj Jain
Date: November 24, 2025
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# Test imports
print("=" * 80)
print("ATTACK API VERIFICATION")
print("=" * 80)

print("\n1. Testing Imports...")
try:
    from src.attacks.fgsm import FGSM, FGSMConfig

    print("   ✅ FGSM imported successfully")
except Exception as e:
    print(f"   ❌ FGSM import failed: {e}")
    sys.exit(1)

try:
    from src.attacks.pgd import PGD, PGDConfig

    print("   ✅ PGD imported successfully")
except Exception as e:
    print(f"   ❌ PGD import failed: {e}")
    sys.exit(1)

try:
    from src.attacks.cw import CarliniWagner, CWConfig

    print("   ✅ C&W imported successfully")
except Exception as e:
    print(f"   ❌ C&W import failed: {e}")
    sys.exit(1)

try:
    from src.attacks.auto_attack import AutoAttack, AutoAttackConfig

    print("   ✅ AutoAttack imported successfully")
except Exception as e:
    print(f"   ❌ AutoAttack import failed: {e}")
    sys.exit(1)

# Test method signatures
print("\n2. Testing Method Signatures...")
import inspect

fgsm_sig = inspect.signature(FGSM.generate)
print(f"   FGSM.generate signature: {fgsm_sig}")
assert "model" in fgsm_sig.parameters, "FGSM.generate missing 'model' parameter"
assert "x" in fgsm_sig.parameters, "FGSM.generate missing 'x' parameter"
assert "y" in fgsm_sig.parameters, "FGSM.generate missing 'y' parameter"
print("   ✅ FGSM.generate has correct signature")

pgd_sig = inspect.signature(PGD.generate)
print(f"   PGD.generate signature: {pgd_sig}")
assert "model" in pgd_sig.parameters, "PGD.generate missing 'model' parameter"
assert "x" in pgd_sig.parameters, "PGD.generate missing 'x' parameter"
assert "y" in pgd_sig.parameters, "PGD.generate missing 'y' parameter"
print("   ✅ PGD.generate has correct signature")

cw_sig = inspect.signature(CarliniWagner.generate)
print(f"   C&W.generate signature: {cw_sig}")
assert "model" in cw_sig.parameters, "C&W.generate missing 'model' parameter"
assert "x" in cw_sig.parameters, "C&W.generate missing 'x' parameter"
assert "y" in cw_sig.parameters, "C&W.generate missing 'y' parameter"
print("   ✅ C&W.generate has correct signature")

# Test PGD step_size parameter
print("\n3. Testing PGD Configuration...")
try:
    # Test with explicit step_size
    config = PGDConfig(
        epsilon=8 / 255,
        num_steps=10,
        step_size=(8 / 255) / 4,  # This is what we added
        device="cpu",
    )
    print(f"   epsilon: {config.epsilon:.6f}")
    print(f"   num_steps: {config.num_steps}")
    print(f"   step_size: {config.step_size:.6f}")
    print("   ✅ PGD accepts step_size parameter")
except Exception as e:
    print(f"   ❌ PGD configuration failed: {e}")
    sys.exit(1)

# Test without step_size (should default to epsilon/4)
try:
    config_default = PGDConfig(
        epsilon=8 / 255,
        num_steps=10,
        device="cpu",
    )
    expected = (8 / 255) / 4
    print(f"   Default step_size: {config_default.step_size:.6f}")
    print(f"   Expected: {expected:.6f}")
    assert abs(config_default.step_size - expected) < 1e-9, "Wrong default step_size"
    print("   ✅ PGD defaults step_size to epsilon/4")
except Exception as e:
    print(f"   ❌ PGD default step_size failed: {e}")
    sys.exit(1)

# Test device handling
print("\n4. Testing Device Handling...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Available device: {device}")

try:
    config = FGSMConfig(epsilon=8 / 255, device=device)
    attack = FGSM(config)
    print(f"   ✅ FGSM initialized on {device}")
except Exception as e:
    print(f"   ❌ FGSM device handling failed: {e}")
    sys.exit(1)

try:
    config = PGDConfig(
        epsilon=8 / 255, num_steps=10, step_size=(8 / 255) / 4, device=device
    )
    attack = PGD(config)
    print(f"   ✅ PGD initialized on {device}")
except Exception as e:
    print(f"   ❌ PGD device handling failed: {e}")
    sys.exit(1)

# Test basic attack execution
print("\n5. Testing Basic Attack Execution...")


class DummyModel(nn.Module):
    """Dummy model for testing."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(3 * 224 * 224, num_classes)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


try:
    model = DummyModel(num_classes=7).to(device)
    model.eval()

    # Create dummy batch
    x = torch.randn(2, 3, 224, 224, device=device)
    y = torch.tensor([0, 1], device=device)

    # Test FGSM
    config = FGSMConfig(epsilon=8 / 255, device=device)
    attack = FGSM(config)
    x_adv = attack.generate(model, x, y)

    assert x_adv.shape == x.shape, f"Wrong output shape: {x_adv.shape}"
    assert x_adv.device == x.device, f"Wrong device: {x_adv.device}"

    # Verify perturbation
    perturbation = (x_adv - x).abs().max().item()
    print(f"   FGSM perturbation: {perturbation:.6f}")
    assert perturbation > 1e-8, "No perturbation generated!"
    assert perturbation <= (8 / 255) + 1e-6, f"Perturbation too large: {perturbation}"
    print("   ✅ FGSM attack executed successfully")

    # Test PGD
    config = PGDConfig(
        epsilon=8 / 255, num_steps=10, step_size=(8 / 255) / 4, device=device
    )
    attack = PGD(config)
    x_adv = attack.generate(model, x, y)

    assert x_adv.shape == x.shape, f"Wrong output shape: {x_adv.shape}"
    assert x_adv.device == x.device, f"Wrong device: {x_adv.device}"

    perturbation = (x_adv - x).abs().max().item()
    print(f"   PGD perturbation: {perturbation:.6f}")
    assert perturbation > 1e-8, "No perturbation generated!"
    assert perturbation <= (8 / 255) + 1e-6, f"Perturbation too large: {perturbation}"
    print("   ✅ PGD attack executed successfully")

except Exception as e:
    print(f"   ❌ Attack execution failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "=" * 80)
print("✅ ALL VERIFICATION TESTS PASSED")
print("=" * 80)
print("\nYour attack implementations are correct and ready for evaluation!")
print("\nNext steps:")
print("1. Ensure baseline models are trained (checkpoints/baseline/seed_42/best.pt)")
print("2. Run quick validation:")
print("   python scripts\\evaluation\\evaluate_baseline_robustness.py --quick")
print("3. Run full evaluation:")
print("   python scripts\\evaluation\\evaluate_baseline_robustness.py --seeds 42")
print("\n" + "=" * 80)
