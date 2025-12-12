"""
Installation Verification Script

Checks if all required packages are installed correctly.
"""

import sys


def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name

    try:
        __import__(import_name)
        print(f"  [OK] {package_name}")
        return True
    except ImportError:
        print(f"  [MISSING] {package_name}")
        return False


def main():
    """Run installation verification."""
    print("=" * 60)
    print("Installation Verification")
    print("=" * 60)

    all_good = True

    # Core dependencies
    print("\n[1/5] Checking Core Dependencies...")
    all_good &= check_package("torch")
    all_good &= check_package("torch-geometric", "torch_geometric")

    # Graph libraries
    print("\n[2/5] Checking Graph Libraries...")
    all_good &= check_package("networkx")
    all_good &= check_package("scipy")

    # ML libraries
    print("\n[3/5] Checking ML Libraries...")
    all_good &= check_package("scikit-learn", "sklearn")
    all_good &= check_package("numpy")

    # Visualization
    print("\n[4/5] Checking Visualization Libraries...")
    all_good &= check_package("matplotlib")
    all_good &= check_package("seaborn")

    # Utilities
    print("\n[5/5] Checking Utility Libraries...")
    all_good &= check_package("tqdm")
    all_good &= check_package("pyyaml", "yaml")
    all_good &= check_package("pandas")

    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("[SUCCESS] All packages installed correctly!")
        print("\nYou're ready to train models!")
        print("\nNext step:")
        print("  python train.py --dataset cora --model gcn --epochs 100")
    else:
        print("[WARNING] Some packages are missing.")
        print("\nPlease install missing packages:")
        print("  pip install <package-name>")
    print("=" * 60)

    return 0 if all_good else 1


if __name__ == '__main__':
    sys.exit(main())
