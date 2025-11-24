# Phase 5.2: Validation & Testing Guide
# =====================================
# How to verify Phase 5.2 meets production-level and A1+ standards

Write-Host "`n" -NoNewline
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 5.2: Validation & Testing Guide" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# =====================================
# 1. COMPREHENSIVE VALIDATION (Recommended)
# =====================================
Write-Host "1. COMPREHENSIVE VALIDATION" -ForegroundColor Green
Write-Host "   Tests: Imports, Error Handling, RQ1, Statistics, Memory, Docs" -ForegroundColor White
Write-Host ""
Write-Host "   python scripts/validation/validate_phase_5_2_comprehensive.py" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 2. QUICK IMPORT CHECK
# =====================================
Write-Host "2. QUICK IMPORT CHECK" -ForegroundColor Green
Write-Host "   Verifies all critical imports work" -ForegroundColor White
Write-Host ""
Write-Host "   python -c `"from scripts.training.train_pgd_at import PGDATTrainer; from scripts.evaluation.evaluate_pgd_at import PGDATEvaluator; print('✅ Imports OK')`"" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 3. UNIT TESTS
# =====================================
Write-Host "3. UNIT TESTS" -ForegroundColor Green
Write-Host "   Runs pytest on Phase 5.2 test suite" -ForegroundColor White
Write-Host ""
Write-Host "   pytest tests/test_phase_5_2_pgd_at.py -v" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 4. RQ1 HYPOTHESIS TEST CHECK
# =====================================
Write-Host "4. RQ1 HYPOTHESIS TEST CHECK" -ForegroundColor Green
Write-Host "   Verify RQ1 cross-site test exists" -ForegroundColor White
Write-Host ""
Write-Host "   python -c `"import re; content = open('scripts/evaluation/evaluate_pgd_at.py').read(); has_rq1 = bool(re.search(r'def.*rq1|def.*cross.*site.*hypothesis', content, re.I)); print('✅ RQ1 test found' if has_rq1 else '❌ RQ1 test MISSING')`"" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 5. STATISTICAL RIGOR CHECK
# =====================================
Write-Host "5. STATISTICAL RIGOR CHECK" -ForegroundColor Green
Write-Host "   Verify advanced statistics (Bonferroni, CI, normality)" -ForegroundColor White
Write-Host ""
Write-Host "   python -c `"content = open('scripts/evaluation/evaluate_pgd_at.py').read(); bonf = 'bonferroni' in content.lower(); ci = 'confidence' in content.lower() and 'interval' in content.lower(); norm = 'shapiro' in content.lower() or 'normaltest' in content.lower(); print(f'Bonferroni: {`"✅`" if bonf else `"❌`"}\nConf Intervals: {`"✅`" if ci else `"❌`"}\nNormality Test: {`"✅`" if norm else `"❌`"}')`"" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 6. ERROR HANDLING CHECK
# =====================================
Write-Host "6. ERROR HANDLING CHECK" -ForegroundColor Green
Write-Host "   Count try-except blocks (need ≥3 per file)" -ForegroundColor White
Write-Host ""
Write-Host "   python -c `"import re; train = open('scripts/training/train_pgd_at.py').read(); eval_s = open('scripts/evaluation/evaluate_pgd_at.py').read(); train_count = len(re.findall(r'\btry:', train)); eval_count = len(re.findall(r'\btry:', eval_s)); print(f'Training: {train_count} handlers {`"✅`" if train_count >= 3 else `"❌`"}\nEvaluation: {eval_count} handlers {`"✅`" if eval_count >= 3 else `"❌`"}')`"" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 7. MEMORY MANAGEMENT CHECK
# =====================================
Write-Host "7. MEMORY MANAGEMENT CHECK" -ForegroundColor Green
Write-Host "   Verify CUDA cache clearing and cleanup" -ForegroundColor White
Write-Host ""
Write-Host "   python -c `"content = open('scripts/evaluation/evaluate_pgd_at.py').read(); cuda = 'torch.cuda.empty_cache()' in content; gc_c = 'gc.collect()' in content; del_s = 'del ' in content; print(f'CUDA clear: {`"✅`" if cuda else `"❌`"}\nGC collect: {`"✅`" if gc_c else `"❌`"}\nExplicit del: {`"✅`" if del_s else `"❌`"}')`"" -ForegroundColor Yellow
Write-Host ""

# =====================================
# 8. DATASET HANDLING CHECK
# =====================================
Write-Host "8. DATASET HANDLING CHECK" -ForegroundColor Green
Write-Host "   Verify all datasets are handled (isic2018/19/20, derm7pt)" -ForegroundColor White
Write-Host ""
Write-Host "   python -c `"content = open('scripts/evaluation/evaluate_pgd_at.py').read().lower(); datasets = ['isic2018', 'isic2019', 'isic2020', 'derm7pt']; results = {d: d in content for d in datasets}; [print(f'{d}: {`"✅`" if v else `"❌`"}') for d, v in results.items()]`"" -ForegroundColor Yellow
Write-Host ""

# =====================================
# VALIDATION GRADE INTERPRETATION
# =====================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "GRADE INTERPRETATION" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Exit Code 0: " -NoNewline
Write-Host "A+ to A1+ " -ForegroundColor Green -NoNewline
Write-Host "- Production-ready & research-complete"
Write-Host ""
Write-Host "Exit Code 2: " -NoNewline
Write-Host "B+ to A-  " -ForegroundColor Yellow -NoNewline
Write-Host "- Code works but missing A1+ requirements"
Write-Host "             (typically missing RQ1 test or advanced stats)"
Write-Host ""
Write-Host "Exit Code 1: " -NoNewline
Write-Host "C to F    " -ForegroundColor Red -NoNewline
Write-Host "- Critical failures (cannot run at all)"
Write-Host ""

# =====================================
# QUICK STATUS CHECK
# =====================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "QUICK STATUS CHECK" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Run this for immediate feedback:" -ForegroundColor White
Write-Host ""
Write-Host "python scripts/validation/validate_phase_5_2_comprehensive.py" -ForegroundColor Yellow
Write-Host ""

# =====================================
# EXPECTED RESULTS FOR A1+ GRADE
# =====================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "EXPECTED RESULTS FOR A1+ GRADE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✅ CRITICAL: All imports work" -ForegroundColor Green
Write-Host "✅ PRODUCTION: Error handling (≥3 try-except per file)" -ForegroundColor Green
Write-Host "✅ PRODUCTION: Dataset handling (all 4 datasets)" -ForegroundColor Green
Write-Host "✅ PRODUCTION: Memory management (CUDA clear, gc.collect)" -ForegroundColor Green
Write-Host "✅ A1+: RQ1 hypothesis test implemented" -ForegroundColor Green
Write-Host "✅ A1+: Bonferroni correction for multiple comparisons" -ForegroundColor Green
Write-Host "✅ A1+: Confidence intervals on effect sizes" -ForegroundColor Green
Write-Host "✅ A1+: Comprehensive documentation (≥250 lines)" -ForegroundColor Green
Write-Host "⚠️  BEST PRACTICE: Normality testing (optional but recommended)" -ForegroundColor Yellow
Write-Host "⚠️  BEST PRACTICE: Power analysis (optional but recommended)" -ForegroundColor Yellow
Write-Host ""

# =====================================
# TROUBLESHOOTING
# =====================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "TROUBLESHOOTING" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "If validation fails:" -ForegroundColor White
Write-Host ""
Write-Host "1. Import errors:" -ForegroundColor Yellow
Write-Host "   - Check src/models/__init__.py has build_model export"
Write-Host "   - Check src/utils/metrics.py exists"
Write-Host "   - Check all 'from src.attacks import' use CarliniWagner (not CW)"
Write-Host ""
Write-Host "2. RQ1 test missing:" -ForegroundColor Yellow
Write-Host "   - Add test_rq1_cross_site_hypothesis() to evaluate_pgd_at.py"
Write-Host "   - Must compare AUROC drops: PGD-AT vs Baseline"
Write-Host "   - See PHASE_5.2_HONEST_ASSESSMENT.md lines 50-85"
Write-Host ""
Write-Host "3. Statistics issues:" -ForegroundColor Yellow
Write-Host "   - Add: from statsmodels.stats.multitest import multipletests"
Write-Host "   - Add Bonferroni correction for 4 datasets"
Write-Host "   - Add confidence intervals on Cohen's d"
Write-Host "   - See PHASE_5.2_HONEST_ASSESSMENT.md lines 90-155"
Write-Host ""
Write-Host "4. Error handling missing:" -ForegroundColor Yellow
Write-Host "   - Wrap checkpoint loading in try-except"
Write-Host "   - Wrap data loading in try-except"
Write-Host "   - Validate file existence with Path.exists()"
Write-Host "   - See PHASE_5.2_HONEST_ASSESSMENT.md lines 200-250"
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
