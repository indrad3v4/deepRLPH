#!/usr/bin/env python3
"""
Submission packaging script for ITEM-008: Strict Packaging (JIT Edition)
Creates submissions/submission.zip with exactly 3 files in flat structure.
"""

import sys
import os
import zipfile
import tempfile
import shutil
from pathlib import Path


def create_submission():
    """Create submission.zip with exactly 3 required files."""

    # REQUIRED: Synchronized with solution.py requirements
    required_files = ['solution.py', 'utils.py', 'model_jit.pt']

    # Create submissions directory if it doesn't exist
    submissions_dir = Path('submissions')
    submissions_dir.mkdir(exist_ok=True)

    # Output zip path
    zip_path = submissions_dir / 'submission.zip'

    # Create a temporary directory for staging
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy each required file to temporary directory
        for file_name in required_files:
            src_path = Path(file_name)
            if not src_path.exists():
                print(f"ERROR: Required file '{file_name}' not found!")
                return False

            dst_path = temp_path / file_name
            shutil.copy2(src_path, dst_path)
            print(f"Added: {file_name}")

        # Create zip file
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files from temp directory (flat structure)
            for file_name in required_files:
                file_path = temp_path / file_name
                # Add to root of zip (no directory structure)
                zipf.write(file_path, arcname=file_name)

        # Verify the zip contents
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zip_contents = zipf.namelist()
            print(f"\nZip created: {zip_path}")
            print(f"Files in zip ({len(zip_contents)}):")
            for file in sorted(zip_contents):
                print(f"  - {file}")

            if len(zip_contents) != 3:
                print(f"ERROR: Expected 3 files, got {len(zip_contents)}!")
                return False

        print("\n✓ Submission package created successfully!")
        print(f"✓ Contains exactly 3 files: {', '.join(sorted(required_files))}")
        return True


if __name__ == "__main__":
    print("Creating submission package for ITEM-008: Strict Packaging")
    print("=" * 60)
    success = create_submission()
    sys.exit(0 if success else 1)