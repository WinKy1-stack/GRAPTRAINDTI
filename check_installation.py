"""
Check installation and system requirements
Run this script to verify all dependencies are installed correctly
"""
import sys
import importlib
from typing import List, Tuple


def check_package(package_name: str, display_name: str = None) -> Tuple[bool, str]:
    """
    Check if a package is installed
    
    Returns:
        (success, version_or_error)
    """
    if display_name is None:
        display_name = package_name
    
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    """Main check function"""
    print("=" * 70)
    print("GraphTransDTI - Installation Check")
    print("=" * 70)
    
    # Core packages
    packages = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('rdkit', 'RDKit'),
        ('Bio', 'BioPython'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('tqdm', 'tqdm'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML')
    ]
    
    results = []
    all_success = True
    
    for package, display in packages:
        success, info = check_package(package, display)
        results.append((display, success, info))
        if not success:
            all_success = False
    
    # Print results
    print("\nPackage Status:")
    print("-" * 70)
    
    for display, success, info in results:
        status = "✓" if success else "✗"
        if success:
            print(f"{status} {display:<25} | Version: {info}")
        else:
            print(f"{status} {display:<25} | ERROR: Not installed")
    
    print("-" * 70)
    
    # Additional checks
    print("\nAdditional Checks:")
    print("-" * 70)
    
    # CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA Available          | Version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA Not Available      | CPU-only mode")
    except:
        print("✗ Cannot check CUDA")
    
    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 8):
        print(f"✓ Python Version          | {py_version}")
    else:
        print(f"✗ Python Version          | {py_version} (requires >= 3.8)")
        all_success = False
    
    print("-" * 70)
    
    # Final summary
    print("\nSummary:")
    if all_success:
        print("✓ All dependencies are installed correctly!")
        print("  You can now run GraphTransDTI.")
        print("\nNext steps:")
        print("  1. Download datasets: See data/DATA_DOWNLOAD_GUIDE.md")
        print("  2. Run training: cd src && python train.py")
        print("  3. Or use notebook: notebooks/Train_GraphTransDTI.ipynb")
    else:
        print("✗ Some dependencies are missing.")
        print("\nTo install missing packages:")
        print("  pip install -r src/requirements.txt")
        print("\nIf PyTorch Geometric fails, try:")
        print("  pip install torch-geometric torch-scatter torch-sparse torch-cluster")
    
    print("=" * 70)
    
    return all_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
