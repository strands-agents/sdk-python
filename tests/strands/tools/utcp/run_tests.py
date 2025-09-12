#!/usr/bin/env python3
"""Test runner for UTCP integration tests.

This script runs all UTCP tests and provides a summary of results.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all UTCP tests and display results."""
    test_dir = Path(__file__).parent

    print("🧪 Running UTCP Integration Tests")
    print("=" * 50)

    # Test files to run
    test_files = [
        "test_utcp_agent_tool.py",
        "test_utcp_client.py",
        "test_utcp_simple.py",
        "test_utcp_integration_e2e.py",
    ]

    results = {}
    total_tests = 0
    total_passed = 0
    total_failed = 0

    for test_file in test_files:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"⚠️  Test file not found: {test_file}")
            continue

        print(f"\n📋 Running {test_file}...")
        print("-" * 30)

        try:
            # Run pytest on the specific file
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                cwd=test_dir.parent.parent.parent.parent,  # Go to project root
            )

            # Parse results
            output_lines = result.stdout.split("\n")
            passed = sum(1 for line in output_lines if " PASSED" in line)
            failed = sum(1 for line in output_lines if " FAILED" in line)
            skipped = sum(1 for line in output_lines if " SKIPPED" in line)

            results[test_file] = {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "return_code": result.returncode,
            }

            total_tests += passed + failed + skipped
            total_passed += passed
            total_failed += failed

            # Display results for this file
            if result.returncode == 0:
                print(f"✅ {test_file}: {passed} passed, {skipped} skipped")
            else:
                print(f"❌ {test_file}: {passed} passed, {failed} failed, {skipped} skipped")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")

        except Exception as e:
            print(f"💥 Error running {test_file}: {e}")
            results[test_file] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)

    for test_file, result in results.items():
        if "error" in result:
            print(f"💥 {test_file}: ERROR - {result['error']}")
        else:
            status = "✅" if result["return_code"] == 0 else "❌"
            print(
                f"{status} {test_file}: {result['passed']} passed, "
                f"{result['failed']} failed, {result['skipped']} skipped"
            )

    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")

    if total_failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print(f"⚠️  {total_failed} tests failed")
        return 1


def run_specific_test(test_name: str):
    """Run a specific test file."""
    test_dir = Path(__file__).parent
    test_path = test_dir / f"test_{test_name}.py"

    if not test_path.exists():
        print(f"❌ Test file not found: test_{test_name}.py")
        return 1

    print(f"🧪 Running test_{test_name}.py")
    print("-" * 30)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_path), "-v"], cwd=test_dir.parent.parent.parent.parent
        )
        return result.returncode
    except Exception as e:
        print(f"💥 Error running test: {e}")
        return 1


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        return run_specific_test(test_name)
    else:
        # Run all tests
        return run_tests()


if __name__ == "__main__":
    sys.exit(main())
