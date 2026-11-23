"""
tests/run_all_tests.py — Run All Tests at Once
================================================
Usage: python tests/run_all_tests.py
Or:    python -m pytest tests/ -v
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all():
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(test_dir, pattern="test_*.py")

    print("\n" + "=" * 65)
    print("  ELDER CARE MONITORING — COMPLETE TEST SUITE")
    print("  Running all unit tests...")
    print("=" * 65 + "\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    print(f"  Tests run:    {result.testsRun}")
    print(f"  Failures:     {len(result.failures)}")
    print(f"  Errors:       {len(result.errors)}")
    print(f"  Skipped:      {len(result.skipped)}")
    print("=" * 65)

    if result.wasSuccessful():
        print("\n[PASS] ALL TESTS PASSED!\n")
        return 0
    else:
        print(f"\n[FAIL] SOME TESTS FAILED ({len(result.failures)} failures, {len(result.errors)} errors)\n")
        return 1


if __name__ == "__main__":
    exit_code = run_all()
    sys.exit(exit_code)
