"""
Quick verification script for Phase 5.4 HPO fix.

This script verifies that the learning_rate KeyError has been fixed.

Author: Viraj Pankaj Jain
Date: November 24, 2025
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.hpo_config import TRADESSearchSpace


def verify_search_space_fix():
    """Verify that the search space properly includes learning_rate."""

    print("=" * 60)
    print("VERIFYING PHASE 5.4 HPO FIX")
    print("=" * 60)

    # Create search space
    search_space = TRADESSearchSpace()

    # Convert to dict (this is what hpo_trainer uses)
    spaces_dict = search_space.to_dict()

    print("\n1. Checking search space dictionary...")
    print(f"   Keys: {list(spaces_dict.keys())}")

    # Verify learning_rate exists
    if "learning_rate" not in spaces_dict:
        print("   ❌ FAILED: 'learning_rate' not in search space")
        return False
    print("   ✅ PASSED: 'learning_rate' found in search space")

    # Verify type is normalized to 'float' (not 'log_float')
    lr_config = spaces_dict["learning_rate"]
    print(f"\n2. Checking learning_rate configuration...")
    print(f"   Type: {lr_config['type']}")
    print(f"   Range: [{lr_config['low']}, {lr_config['high']}]")
    print(f"   Log scale: {lr_config['log']}")

    if lr_config["type"] not in ["float", "int"]:
        print(f"   ❌ FAILED: Type should be 'float', got '{lr_config['type']}'")
        return False
    print("   ✅ PASSED: Type is properly normalized")

    # Verify log flag is set
    if not lr_config["log"]:
        print("   ⚠️  WARNING: log=False, but learning_rate should use log scale")
    else:
        print("   ✅ PASSED: Log scale enabled")

    # Check all required parameters
    required_params = ["beta", "epsilon", "learning_rate"]
    print(f"\n3. Checking required parameters...")
    missing = [p for p in required_params if p not in spaces_dict]
    if missing:
        print(f"   ❌ FAILED: Missing parameters: {missing}")
        return False
    print(f"   ✅ PASSED: All required parameters present")

    # Display full configuration
    print(f"\n4. Full search space configuration:")
    for param, config in spaces_dict.items():
        print(f"   {param}:")
        print(f"     - type: {config['type']}")
        if config.get("choices"):
            print(f"     - choices: {config['choices']}")
        else:
            print(f"     - range: [{config['low']}, {config['high']}]")
        print(f"     - log: {config['log']}")

    print("\n" + "=" * 60)
    print("✅ ALL CHECKS PASSED - FIX VERIFIED")
    print("=" * 60)
    print("\nYou can now run HPO with:")
    print("  python scripts/run_hpo_study.py --quick-test")

    return True


if __name__ == "__main__":
    try:
        success = verify_search_space_fix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
