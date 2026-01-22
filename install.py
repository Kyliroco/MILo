import os
import re
import argparse


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup the environment')

    parser.add_argument('--cuda_version', type=str, default='11.8',
                        help='CUDA version to use (e.g., 11.8, 12.1, 12.2, 12.4, 12.6...)')
    args = parser.parse_args()

    print(f"[INFO] Installing environment...")

    # Map CUDA version to PyTorch CUDA version
    try:
        pytorch_cuda_version = get_pytorch_cuda_version(args.cuda_version)
    except ValueError as e:
        print(f"[ERROR] {e}")
        exit(1)

    if args.cuda_version != pytorch_cuda_version:
        print(f"[INFO] CUDA {args.cuda_version} detected. Using PyTorch built for CUDA {pytorch_cuda_version} (backward compatible).")
        print(f"[INFO] Make sure your CUDA {args.cuda_version} paths are set correctly:")
        print(f"       export CPATH=/usr/local/cuda-{args.cuda_version}/targets/x86_64-linux/include:$CPATH")
        print(f"       export LD_LIBRARY_PATH=/usr/local/cuda-{args.cuda_version}/targets/x86_64-linux/lib:$LD_LIBRARY_PATH")
        print(f"       export PATH=/usr/local/cuda-{args.cuda_version}/bin:$PATH")

    # Install torch
    print(f"[INFO] Installing torch...")
    os.system(f"conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda={pytorch_cuda_version} mkl=2023.1.0 -c pytorch -c nvidia")
    print(f"[INFO] Torch installed.")
    
    # Install requirements
    print(f"[INFO] Installing requirements...")
    os.system(f"pip install -r requirements.txt")
    print(f"[INFO] Requirements installed.")
    
    # Install submodules
    print(f"[INFO] Installing Mini-Splatting2 rasterizer...")
    os.system(f"pip install submodules/diff-gaussian-rasterization_ms")
    print("[INFO] Mini-Splatting2 rasterizer installed.")

    print(f"[INFO] Installing RaDe-GS rasterizer...")
    os.system(f"pip install submodules/diff-gaussian-rasterization")
    print("[INFO] RaDe-GS rasterizer installed.")
    
    print(f"[INFO] Installing GOF rasterizer...")
    os.system(f"pip install submodules/diff-gaussian-rasterization_gof")
    print("[INFO] GOF rasterizer installed.")
    
    print(f"[INFO] Installing Simple KNN...")
    os.system(f"pip install submodules/simple-knn")
    print("[INFO] Simple KNN installed.")
    
    print(f"[INFO] Installing Fused SSIM...")
    os.system(f"pip install submodules/fused-ssim")
    print("[INFO] Fused SSIM installed.")
    
    print(f"[INFO] Installing Triangulation...")
    os.chdir("submodules/tetra_triangulation/")
    os.system(f"conda install -y cmake")
    os.system(f"conda install -y conda-forge::gmp")
    os.system(f"conda install -y conda-forge::cgal")
    # WARNING: CUDA paths must be set before running cmake
    os.system(f"cmake .")
    os.system(f"make")
    os.system(f"pip install -e .")
    os.chdir("../../")
    print("[INFO] Triangulation installed.")
    
    print(f"[INFO] Installing Nvdiffrast...")
    os.chdir("submodules/nvdiffrast/")
    os.system(f"pip install -e .")
    os.chdir("../../")
    print("[INFO] Nvdiffrast installed.")
    
    print(f"[INFO] Installation complete.")
    