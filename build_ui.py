#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
import shutil

def main():
    """Build the UI using pnpm."""
    ui_dir = Path(__file__).parent / "moatless-ui"
    moatless_api_dir = Path(__file__).parent / "moatless_api"
    moatless_api_pkg = moatless_api_dir  # Package directory
    dist_dir = moatless_api_pkg / "ui/dist"
    
    if not ui_dir.exists():
        print("UI directory not found")
        return
    
    print("Building UI...")
    try:
        # Install dependencies
        subprocess.run(["bun", "install"], cwd=ui_dir, check=True)
        # Build UI
        subprocess.run(["bun", "run", "build"], cwd=ui_dir, check=True)
        
        # Create package directories and __init__.py files
        moatless_api_dir.mkdir(parents=True, exist_ok=True)
        moatless_api_pkg.mkdir(parents=True, exist_ok=True)
        (moatless_api_pkg / "ui").mkdir(parents=True, exist_ok=True)
        
        # Create package __init__.py files
        (moatless_api_dir / "__init__.py").write_text('"""Package for Moatless API UI files."""\n')
        (moatless_api_pkg / "__init__.py").write_text('"""Moatless API package."""\n')
        (moatless_api_pkg / "ui" / "__init__.py").write_text('"""UI package."""\n')
        
        # Create setup.py
        setup_py = moatless_api_dir / "setup.py"
        setup_py.write_text("""
from setuptools import setup, find_packages

setup(
    name="moatless_api",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "moatless_api": ["ui/dist/**/*"],
    },
)
""")
        
        # Create MANIFEST.in
        manifest = moatless_api_dir / "MANIFEST.in"
        manifest.write_text("recursive-include moatless_api/ui/dist *")
        
        # Move built files to package directory
        if dist_dir.exists():
            shutil.rmtree(dist_dir)
        dist_dir.parent.mkdir(parents=True, exist_ok=True)
        print(f"Moving built files from {ui_dir / 'dist'} to {dist_dir}")
        shutil.move(str(ui_dir / "dist"), str(dist_dir))
        
        print("UI build complete")
    except subprocess.CalledProcessError as e:
        print(f"Failed to build UI: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("bun not found. Please install bun to build the UI")
        sys.exit(1)

if __name__ == "__main__":
    main() 