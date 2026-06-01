"""
install.py — Smart installer that handles SHAP's Windows build issue.

Run this INSTEAD of: pip install -r requirements.txt

Usage:
    python install.py
"""

import subprocess
import sys
import platform


def run(cmd):
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    py = sys.executable
    print("=" * 60)
    print("  Retail Churn — Dependency Installer")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  OS     : {platform.system()} {platform.release()}")
    print("=" * 60)

    # Step 1 — upgrade pip + wheel + setuptools (needed for SHAP)
    print("\n[1/3] Upgrading pip, wheel, setuptools...")
    run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])

    # Step 2 — install everything except SHAP
    core = [
        "pandas>=2.0.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
    ]
    print("\n[2/3] Installing core packages...")
    ok = run([py, "-m", "pip", "install"] + core)
    if not ok:
        print("  ERROR: Core install failed. Check your Python/pip setup.")
        sys.exit(1)

    # Step 3 — try SHAP, with fallback strategies
    print("\n[3/3] Installing SHAP...")

    # Try 1: pre-built binary wheel only (fastest, no compiler needed)
    ok = run([py, "-m", "pip", "install", "shap", "--only-binary", ":all:"])

    if not ok:
        print("  Pre-built wheel not found — trying specific version 0.43.0...")
        # Try 2: older version with known Windows wheels
        ok = run([py, "-m", "pip", "install", "shap==0.43.0", "--only-binary", ":all:"])

    if not ok:
        print("  Trying 0.42.1...")
        ok = run([py, "-m", "pip", "install", "shap==0.42.1", "--only-binary", ":all:"])

    if not ok:
        print("\n  SHAP could not be installed automatically.")
        print("\n  Manual fix options:")
        print("  A) Install Visual C++ Build Tools, then: pip install shap")
        print("     https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print()
        print("  B) Use conda (recommended for Windows):")
        print("     conda install -c conda-forge shap")
        print()
        print("  C) The pipeline will still run without SHAP")
        print("     (SHAP plots will be skipped automatically).")
        print()
        print("  Continuing install without SHAP...")
    else:
        print("  SHAP installed successfully.")

    print("\n" + "=" * 60)
    print("  Installation complete.")
    print("  Next: python pipelines/run_pipeline.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
