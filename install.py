import os
import re
import sys
import shutil
import argparse
import subprocess


def parse_cuda_version(version_str):
    """Parse CUDA version string and return (major, minor) tuple."""
    match = re.match(r'^(\d+)\.(\d+)$', version_str)
    if not match:
        raise ValueError(f"Invalid CUDA version format: {version_str}. Expected format: X.Y (e.g., 11.8, 12.1)")
    return int(match.group(1)), int(match.group(2))


def get_pytorch_cuda_version(cuda_version):
    """
    Map system CUDA version to compatible PyTorch CUDA version.
    PyTorch 2.3.1 supports: pytorch-cuda=11.8 and pytorch-cuda=12.1
    CUDA is backward compatible within major versions.
    """
    major, minor = parse_cuda_version(cuda_version)

    if major == 11:
        if minor < 8:
            print(f"[WARNING] CUDA {cuda_version} is older than 11.8. Using pytorch-cuda=11.8 (may have compatibility issues).")
        return '11.8'
    elif major >= 12:
        # All CUDA 12.x versions are backward compatible with PyTorch built for 12.1
        return '12.1'
    else:
        raise ValueError(f"CUDA {cuda_version} is not supported. Minimum supported version is 11.x")


def run_command(cmd, description, check=True, use_bash=False):
    """Run a command and handle errors."""
    print(f"[INFO] {description}...")

    if use_bash:
        # Use bash to run conda commands (needed for conda init)
        result = subprocess.run(cmd, shell=True, executable='/bin/bash')
    else:
        result = subprocess.run(cmd, shell=True)

    if check and result.returncode != 0:
        print(f"[ERROR] Failed: {description}")
        return False
    return True


def check_torch_installed():
    """Check if PyTorch is installed and working."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"[INFO] PyTorch version: {result.stdout.strip()}")
            return True
    except Exception:
        pass
    return False


def find_conda():
    """Find conda executable path."""
    # Check if conda is in PATH
    conda_path = shutil.which('conda')
    if conda_path:
        return conda_path

    # Common conda locations
    home = os.path.expanduser("~")
    possible_paths = [
        os.path.join(home, "anaconda3", "bin", "conda"),
        os.path.join(home, "miniconda3", "bin", "conda"),
        os.path.join(home, "conda", "bin", "conda"),
        "/opt/conda/bin/conda",
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            return path

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')

    parser.add_argument('--cuda_version', type=str, default='11.8',
                        help='CUDA version to use (e.g., 11.8, 12.1, 12.2, 12.4, 12.6...)')
    parser.add_argument('--skip_pytorch', action='store_true',
                        help='Skip PyTorch installation (if already installed)')
    args = parser.parse_args()

    print(f"[INFO] Installing environment...")
    print(f"[INFO] Python executable: {sys.executable}")

    # Map CUDA version to PyTorch CUDA version
    try:
        pytorch_cuda_version = get_pytorch_cuda_version(args.cuda_version)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Convert 11.8 -> cu118, 12.1 -> cu121
    cuda_suffix = f"cu{pytorch_cuda_version.replace('.', '')}"

    if args.cuda_version != pytorch_cuda_version:
        print(f"[INFO] CUDA {args.cuda_version} detected. Using PyTorch built for CUDA {pytorch_cuda_version} (backward compatible).")
        print(f"[INFO] Make sure your CUDA {args.cuda_version} paths are set correctly:")
        print(f"       export CPATH=/usr/local/cuda-{args.cuda_version}/targets/x86_64-linux/include:$CPATH")
        print(f"       export LD_LIBRARY_PATH=/usr/local/cuda-{args.cuda_version}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH")
        print(f"       export PATH=/usr/local/cuda-{args.cuda_version}/bin:$PATH")

    # Install PyTorch using pip (more reliable than conda in subprocesses)
    if args.skip_pytorch:
        print(f"[INFO] Skipping PyTorch installation (--skip_pytorch flag set)")
    else:
        print(f"[INFO] Installing PyTorch with CUDA {pytorch_cuda_version} using pip...")
        pytorch_install_cmd = (
            f"{sys.executable} -m pip install "
            f"torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "
            f"--index-url https://download.pytorch.org/whl/{cuda_suffix}"
        )
        if not run_command(pytorch_install_cmd, "Installing PyTorch"):
            print("[ERROR] Failed to install PyTorch. Please install it manually:")
            print(f"        pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/{cuda_suffix}")
            sys.exit(1)

    # Verify PyTorch installation
    if not check_torch_installed():
        print("[ERROR] PyTorch is not installed correctly. Please install it manually before continuing.")
        sys.exit(1)

    # Install requirements
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")

    # Install submodules (these require PyTorch to be installed first)
    print("[INFO] Installing CUDA extensions (this may take a while)...")

    submodules = [
        ("submodules/diff-gaussian-rasterization_ms", "Mini-Splatting2 rasterizer"),
        ("submodules/diff-gaussian-rasterization", "RaDe-GS rasterizer"),
        ("submodules/diff-gaussian-rasterization_gof", "GOF rasterizer"),
        ("submodules/simple-knn", "Simple KNN"),
        ("submodules/fused-ssim", "Fused SSIM"),
    ]

    for path, name in submodules:
        if not run_command(f"{sys.executable} -m pip install {path}", f"Installing {name}", check=False):
            print(f"[WARNING] Failed to install {name}. You may need to install it manually.")

    # Install Triangulation dependencies
    print("[INFO] Installing Triangulation dependencies...")

    # Try to find conda for installing cmake, gmp, cgal
    conda_path = find_conda()
    if conda_path:
        print(f"[INFO] Found conda at: {conda_path}")
        # Use full path to conda
        run_command(f"{conda_path} install -y cmake", "Installing cmake", check=False, use_bash=True)
        run_command(f"{conda_path} install -y -c conda-forge gmp", "Installing gmp", check=False, use_bash=True)
        run_command(f"{conda_path} install -y -c conda-forge cgal", "Installing cgal", check=False, use_bash=True)
    else:
        print("[WARNING] conda not found. Please install cmake, gmp, and cgal manually:")
        print("          conda install cmake")
        print("          conda install -c conda-forge gmp")
        print("          conda install -c conda-forge cgal")

    # Build tetra_triangulation
    original_dir = os.getcwd()
    tetra_dir = os.path.join(original_dir, "submodules/tetra_triangulation")

    if os.path.isdir(tetra_dir):
        os.chdir(tetra_dir)
        run_command("cmake .", "Running cmake for Triangulation", check=False)
        run_command("make", "Building Triangulation", check=False)
        run_command(f"{sys.executable} -m pip install -e .", "Installing Triangulation", check=False)
        os.chdir(original_dir)
    else:
        print(f"[WARNING] Triangulation directory not found: {tetra_dir}")

    # Install Nvdiffrast
    nvdiffrast_dir = os.path.join(original_dir, "submodules/nvdiffrast")
    if os.path.isdir(nvdiffrast_dir):
        os.chdir(nvdiffrast_dir)
        run_command(f"{sys.executable} -m pip install .", "Installing Nvdiffrast", check=False)
        os.chdir(original_dir)
    else:
        print(f"[WARNING] Nvdiffrast directory not found: {nvdiffrast_dir}")

    print("")
    print("=" * 60)
    print("[INFO] Installation complete!")
    print("=" * 60)
    print("")
    print("If you encountered errors with CUDA extensions, make sure:")
    print(f"  1. Your CUDA {args.cuda_version} paths are set correctly:")
    print(f"     export CPATH=/usr/local/cuda-{args.cuda_version}/targets/x86_64-linux/include:$CPATH")
    print(f"     export LD_LIBRARY_PATH=/usr/local/cuda-{args.cuda_version}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH")
    print(f"     export PATH=/usr/local/cuda-{args.cuda_version}/bin:$PATH")
    print("  2. nvcc is available: nvcc --version")
    print("  3. You have a compatible GPU")
    print("")
