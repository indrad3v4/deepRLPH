#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_submission.py - BE-007.2: Automated Submission Packager

Creates competition submission.zip with:
- solution.py
- models/final/model.onnx
- Any other required files

Validates before packaging and tests the zip contents.
"""

import sys
import logging
import zipfile
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SubmissionPackager")


class SubmissionPackager:
    """Strictly packages ONLY the 3 allowed files into a flat ZIP."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        # STRICT WHITELIST - EXACTLY 3 FILES
        self.required_files = [
            "solution.py",
            "utils.py",
            "model_jit.pt"
        ]
        self.optional_files = []  # ABSOLUTELY NO OPTIONAL FILES ALLOWED

    def create_submission(self, output_name: str = None) -> str:
        """
        Create submission zip file enforcing flat structure.
        """
        logger.info("\n📦 Creating STRICT submission package...")

        if output_name is None:
            output_name = "submission.zip"

        submissions_dir = self.project_root / "submissions"
        submissions_dir.mkdir(exist_ok=True)
        zip_path = submissions_dir / output_name

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.required_files:
                full_path = self.project_root / file_path
                if not full_path.exists():
                    logger.error(f"CRITICAL ERROR: Required file {file_path} is missing!")
                    raise FileNotFoundError(f"Cannot package. Missing required file: {file_path}")

                # ENFORCE FLAT STRUCTURE
                zf.write(full_path, arcname=full_path.name)
                logger.info(f"  ✓ Added: {full_path.name} ({full_path.stat().st_size / 1024:.1f} KB)")

        self._verify_zip(zip_path)

        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"\n✅ Submission created: {zip_path.name} ({zip_size_mb:.2f} MB)")
        logger.info(f"   Location: {zip_path}")

        return str(zip_path)

    def _validate(self) -> bool:
        """Skip pre-validation; handled strictly during zipping."""
        return True

    def _verify_zip(self, zip_path: Path) -> None:
        """Verify zip contains EXACTLY the required files, no directories."""
        logger.info("\nVerifying zip contents...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            expected_names = sorted([Path(f).name for f in self.required_files])
            actual_names = sorted(file_list)

            if actual_names != expected_names:
                raise ValueError(f"ZIP payload contaminated! Expected {expected_names}, got {actual_names}")
            logger.info("  ✓ Zip payload structure verified successfully.")

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_submission.py <project_root> [output_name.zip]")
        print("Example: python scripts/make_submission.py .")
        print("Example: python scripts/make_submission.py . my_submission.zip")
        sys.exit(1)
    
    project_root = Path(sys.argv[1])
    output_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not project_root.exists():
        logger.error(f"Project root not found: {project_root}")
        sys.exit(1)
    
    try:
        packager = SubmissionPackager(project_root)
        zip_path = packager.create_submission(output_name)
        logger.info(f"\n🎉 Ready to submit: {zip_path}")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Failed to create submission: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
