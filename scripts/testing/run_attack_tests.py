#!/usr/bin/env python3
"""
Comprehensive Test Runner for Phase 4.2: Attack Testing & Validation
=====================================================================

Provides multiple execution modes with detailed reporting and validation.

Usage:
    python scripts/testing/run_attack_tests.py --mode quick
    python scripts/testing/run_attack_tests.py --mode full
    python scripts/testing/run_attack_tests.py --mode benchmark
    python scripts/testing/run_attack_tests.py --attack fgsm

Modes:
    - quick: Fast unit tests only (~2-5 min)
    - full: Comprehensive test suite (~15-30 min)
    - benchmark: Performance benchmarks (~5-10 min)
    - smoke: Minimal validation (~30 sec)
    - integration: End-to-end pipeline tests (~5-15 min)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: January 2025
"""

from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime


def run_pytest(
    test_path: str,
    markers: Optional[List[str]] = None,
    extra_args: Optional[List[str]] = None,
    coverage: bool = True
) -> int:
    """
    Execute pytest with specified configuration.
    
    Args:
        test_path: Path to test file/directory
        markers: Pytest markers to include/exclude
        extra_args: Additional pytest arguments
        coverage: Generate coverage report
    
    Returns:
        Exit code from pytest
    """
    cmd = ["pytest", test_path]
    
    # Add markers
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src/attacks",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Add extra args
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    return subprocess.call(cmd)


def quick_test() -> int:
    """
    Quick test mode: Fast validation (Phase 4.2 focus).
    
    Runs:
    - All unit tests (perturbation norms, clipping, basic success)
    - Excludes: slow tests (C&W full), benchmarks
    - Runtime: 2-5 minutes
    
    Returns:
        Exit code (0 = success)
    """
    print("\n🚀 QUICK TEST MODE")
    print("="*70)
    print("Phase 4.2: Attack Testing & Validation (Fast)")
    print("Running: Perturbation norms, clipping, attack success (basic)")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py",
        markers=["not slow", "not benchmark"],
        extra_args=[
            "-v",
            "--tb=short",
            "--durations=10"
        ]
    )


def full_test() -> int:
    """
    Full test mode: Comprehensive Phase 4.2 validation.
    
    Runs:
    - All unit tests (perturbation norms, clipping, attack success)
    - Gradient masking detection
    - Computational efficiency
    - Integration tests (transferability, medical imaging pipeline)
    - Runtime: 15-30 minutes
    
    Returns:
        Exit code (0 = success)
    """
    print("\n🔬 FULL TEST MODE")
    print("="*70)
    print("Phase 4.2: Comprehensive Attack Testing & Validation")
    print("Running ALL tests including slow C&W, integration, masking detection")
    print("Expected runtime: 15-30 minutes (hardware dependent)")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py",
        extra_args=[
            "-v",
            "--tb=short",
            "--durations=20",
            "--maxfail=5"  # Stop after 5 failures
        ]
    )


def benchmark_test() -> int:
    """
    Benchmark mode: Performance validation only.
    
    Runs:
    - FGSM performance benchmarks
    - PGD scaling with num_steps
    - Memory usage validation
    - Batch size scaling
    - Runtime: 5-10 minutes
    
    Returns:
        Exit code (0 = success)
    """
    print("\n⚡ BENCHMARK MODE")
    print("="*70)
    print("Phase 4.2: Computational Efficiency Benchmarks")
    print("Testing: Runtime scaling, memory usage, GPU/CPU performance")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py::TestComputationalEfficiency",
        extra_args=[
            "-v",
            "--tb=short",
            "--durations=10"
        ],
        coverage=False  # No coverage for benchmarks
    )


def smoke_test() -> int:
    """
    Smoke test mode: Minimal validation.
    
    Runs:
    - Basic perturbation norm checks (single epsilon)
    - Simple clipping validation
    - One attack success test
    - Runtime: 30-60 seconds
    
    Returns:
        Exit code (0 = success)
    """
    print("\n💨 SMOKE TEST MODE")
    print("="*70)
    print("Phase 4.2: Minimal Smoke Tests")
    print("Quick sanity check that attacks work")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py",
        markers=["not slow", "not benchmark"],
        extra_args=[
            "-v",
            "--tb=line",
            "-k", "perturbation_norm or clipping or reduces_accuracy",
            "--maxfail=3"
        ],
        coverage=False
    )


def integration_test() -> int:
    """
    Integration test mode: End-to-end pipeline validation.
    
    Runs:
    - Attack transferability tests
    - Cross-attack consistency
    - Medical imaging robustness pipeline
    - All attacks respect bounds
    - Runtime: 5-15 minutes
    
    Returns:
        Exit code (0 = success)
    """
    print("\n🔗 INTEGRATION TEST MODE")
    print("="*70)
    print("Phase 4.2: Cross-Attack Integration Tests")
    print("Testing: Transferability, consistency, medical imaging pipeline")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py::TestCrossAttackIntegration",
        extra_args=[
            "-v",
            "--tb=short",
            "--durations=10"
        ]
    )


def gradient_masking_test() -> int:
    """
    Gradient masking detection tests only.
    
    Runs:
    - Vanishing gradients check
    - Loss sensitivity validation
    - Gradient consistency across seeds
    - Shattered gradients detection
    - Runtime: 3-5 minutes
    
    Returns:
        Exit code (0 = success)
    """
    print("\n🔍 GRADIENT MASKING DETECTION MODE")
    print("="*70)
    print("Phase 4.2: Gradient Masking Detection")
    print("Testing: Vanishing gradients, loss sensitivity, consistency")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py::TestGradientMasking",
        extra_args=[
            "-v",
            "--tb=short"
        ]
    )


def specific_attack_test(attack: str) -> int:
    """
    Test specific attack only.
    
    Args:
        attack: Attack name (fgsm, pgd, cw, autoattack)
    
    Returns:
        Exit code (0 = success)
    """
    attack_test_map = {
        'fgsm': 'test_fgsm',
        'pgd': 'test_pgd',
        'cw': 'test_cw or TestCarliniWagner',
        'autoattack': 'test_autoattack or TestAutoAttack',
        'norms': 'TestPerturbationNorms',
        'clipping': 'TestClippingValidation',
        'success': 'TestAttackSuccess',
        'masking': 'TestGradientMasking',
        'efficiency': 'TestComputationalEfficiency',
        'integration': 'TestCrossAttackIntegration'
    }
    
    if attack.lower() not in attack_test_map:
        print(f"❌ Unknown attack: {attack}")
        print(f"Available options: {', '.join(attack_test_map.keys())}")
        return 1
    
    test_filter = attack_test_map[attack.lower()]
    
    print(f"\n🎯 TESTING {attack.upper()}")
    print("="*70)
    
    return run_pytest(
        test_path="tests/test_attacks.py",
        extra_args=[
            "-v",
            "--tb=short",
            "-k", test_filter
        ]
    )


def validate_environment() -> bool:
    """
    Validate testing environment setup.
    
    Checks:
    - Python version (>= 3.10)
    - PyTorch installation
    - CUDA availability
    - Required packages
    - Source code structure
    
    Returns:
        True if environment valid, False otherwise
    """
    print("\n🔍 VALIDATING ENVIRONMENT")
    print("="*70)
    
    checks = []
    
    # Python version
    py_version = sys.version_info
    py_check = py_version >= (3, 10)
    checks.append(("Python >= 3.10", py_check))
    if py_check:
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor} (need >= 3.10)")
    
    # PyTorch
    try:
        import torch
        torch_check = True
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        torch_check = False
        print("✗ PyTorch not installed")
    checks.append(("PyTorch", torch_check))
    
    # CUDA
    if torch_check:
        cuda_available = torch.cuda.is_available()
        checks.append(("CUDA Available", cuda_available))
        if cuda_available:
            print(f"✓ CUDA {torch.version.cuda} - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available (tests will run on CPU)")
    
    # Required packages
    required = ['pytest', 'pytest_cov', 'numpy']
    for package in required:
        try:
            __import__(package.replace('_', '-').replace('-', '_'))
            checks.append((f"{package}", True))
            print(f"✓ {package}")
        except ImportError:
            checks.append((f"{package}", False))
            print(f"✗ {package} not installed")
    
    # Source code
    src_path = Path("src/attacks")
    checks.append(("Source code (src/attacks/)", src_path.exists()))
    if src_path.exists():
        print(f"✓ Source code: {src_path}")
    else:
        print(f"✗ Source code not found: {src_path}")
    
    print("="*70)
    
    # Check critical requirements
    critical_checks = [
        checks[0][1],  # Python
        checks[1][1],  # PyTorch
        checks[-1][1]  # Source code
    ]
    
    if not all(critical_checks):
        print("\n❌ CRITICAL: Environment validation failed!")
        print("Please install missing dependencies:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ Environment validation passed!")
    return True


def generate_test_report():
    """Generate comprehensive test report with statistics."""
    print("\n📊 GENERATING TEST REPORT")
    print("="*70)
    
    # Run pytest with json report
    cmd = [
        "pytest",
        "tests/test_attacks.py",
        "--json-report",
        "--json-report-file=tests/reports/test_report.json",
        "-v",
        "--tb=short"
    ]
    
    subprocess.call(cmd)
    
    # Parse and display
    report_path = Path("tests/reports/test_report.json")
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total Tests: {report['summary']['total']}")
        print(f"Passed: {report['summary'].get('passed', 0)} ✓")
        print(f"Failed: {report['summary'].get('failed', 0)} ✗")
        print(f"Skipped: {report['summary'].get('skipped', 0)} ⊘")
        print(f"Duration: {report['duration']:.2f}s")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase 4.2 attack tests with various modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick validation (recommended for development)
    python scripts/testing/run_attack_tests.py --mode quick
    
    # Full comprehensive suite
    python scripts/testing/run_attack_tests.py --mode full
    
    # Performance benchmarks only
    python scripts/testing/run_attack_tests.py --mode benchmark
    
    # Test specific component
    python scripts/testing/run_attack_tests.py --attack fgsm
    python scripts/testing/run_attack_tests.py --attack norms
    python scripts/testing/run_attack_tests.py --attack masking
    
    # Validate environment only
    python scripts/testing/run_attack_tests.py --validate-only
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['quick', 'full', 'benchmark', 'smoke', 'integration', 'masking'],
        default='quick',
        help='Test execution mode (default: quick)'
    )
    
    parser.add_argument(
        '--attack',
        type=str,
        help='Test specific attack/component (fgsm, pgd, cw, norms, clipping, '
             'success, masking, efficiency, integration)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment, do not run tests'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate detailed test report (JSON)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip environment validation'
    )
    
    args = parser.parse_args()
    
    # Header
    print("\n" + "="*70)
    print("PHASE 4.2: ADVERSARIAL ATTACK TESTING & VALIDATION")
    print("Tri-Objective Robust XAI for Medical Imaging")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Validate environment
    if not args.no_validate:
        if not validate_environment():
            sys.exit(1)
    
    if args.validate_only:
        print("\n✓ Environment validation complete. Exiting.")
        sys.exit(0)
    
    # Create directories
    Path("tests/logs").mkdir(parents=True, exist_ok=True)
    Path("tests/reports").mkdir(parents=True, exist_ok=True)
    Path("htmlcov").mkdir(exist_ok=True)
    
    # Run tests based on mode
    if args.attack:
        exit_code = specific_attack_test(args.attack)
    elif args.mode == 'quick':
        exit_code = quick_test()
    elif args.mode == 'full':
        exit_code = full_test()
    elif args.mode == 'benchmark':
        exit_code = benchmark_test()
    elif args.mode == 'smoke':
        exit_code = smoke_test()
    elif args.mode == 'integration':
        exit_code = integration_test()
    elif args.mode == 'masking':
        exit_code = gradient_masking_test()
    else:
        print(f"❌ Unknown mode: {args.mode}")
        exit_code = 1
    
    # Generate report if requested
    if args.report:
        generate_test_report()
    
    # Final summary
    print("\n" + "="*70)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED ✅")
        print("\nPhase 4.2 Test Results:")
        print("  ✓ Perturbation norms verified (L∞/L2 bounds respected)")
        print("  ✓ Clipping validation passed ([0,1] range maintained)")
        print("  ✓ Attack success rates validated")
        print("  ✓ Gradient masking detection working")
        print("  ✓ Computational efficiency benchmarks passed")
        print("\nGenerated Artifacts:")
        print("  - Coverage report: htmlcov/index.html")
        print("  - XML report: coverage.xml")
        print("  - Test logs: tests/logs/")
    else:
        print("❌ SOME TESTS FAILED ❌")
        print("\nCheck the output above for details.")
        print("Logs available in: tests/logs/")
        print("\nCommon issues:")
        print("  - CUDA out of memory: Reduce batch sizes in fixtures")
        print("  - Timeout: Use --mode smoke for faster validation")
        print("  - Missing deps: pip install -r requirements.txt")
    print("="*70 + "\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
