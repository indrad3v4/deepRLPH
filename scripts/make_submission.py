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
    """Package submission files into competition-ready zip"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.required_files = [
            "solution.py",
            "models/final/model.onnx",
        ]
        self.optional_files = [
            "requirements.txt",
            "README.md",
        ]
    
    def create_submission(self, output_name: str = None) -> str:
        """
        Create submission zip file.
        
        Args:
            output_name: Optional custom zip name (default: submission_{timestamp}.zip)
            
        Returns:
            Path to created zip file
        """
        logger.info("\n📦 Creating submission package...")
        
        # Step 1: Validate first
        if not self._validate():
            raise ValueError("Validation failed, cannot create submission")
        
        # Step 2: Determine output path
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"submission_{timestamp}.zip"
        
        submissions_dir = self.project_root / "submissions"
        submissions_dir.mkdir(exist_ok=True)
        
        zip_path = submissions_dir / output_name
        
        # Step 3: Create zip
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add required files
            for file_path in self.required_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    # Store with flattened name (solution.py, model.onnx)
                    arcname = full_path.name
                    zf.write(full_path, arcname=arcname)
                    logger.info(f"  ✓ Added: {arcname} ({full_path.stat().st_size / 1024:.1f} KB)")
            
            # Add optional files if present
            for file_path in self.optional_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    zf.write(full_path, arcname=full_path.name)
                    logger.info(f"  ✓ Added (optional): {full_path.name}")
        
        # Step 4: Verify zip
        self._verify_zip(zip_path)
        
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"\n✅ Submission created: {zip_path.name} ({zip_size_mb:.2f} MB)")
        logger.info(f"   Location: {zip_path}")
        
        return str(zip_path)
    
    def _validate(self) -> bool:
        """Run validation before packaging"""
        logger.info("Running pre-package validation...")
        
        # Use validator if available
        try:
            from validate_submission import SubmissionValidator
            validator = SubmissionValidator(self.project_root)
            return validator.validate()
        except ImportError:
            logger.warning("validate_submission.py not found, skipping validation")
            
            # Basic checks
            for file_path in self.required_files:
                full_path = self.project_root / file_path
                if not full_path.exists():
                    logger.error(f"Required file missing: {file_path}")
                    return False
            
            return True
    
    def _verify_zip(self, zip_path: Path) -> None:
        """Verify zip contents are correct"""
        logger.info("\nVerifying zip contents...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            file_list = zf.namelist()
            
            # Check required files present
            required_names = [Path(f).name for f in self.required_files]
            for req in required_names:
                if req not in file_list:
                    logger.warning(f"  ⚠️  {req} not found in zip")
                else:
                    logger.info(f"  ✓ {req}")
            
            # Test extraction to temp directory
            with tempfile.TemporaryDirectory() as tmpdir:
                zf.extractall(tmpdir)
                logger.info(f"  ✓ Extraction test passed")
                
                # Test importing solution.py
                solution_path = Path(tmpdir) / "solution.py"
                if solution_path.exists():
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("solution", solution_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            if hasattr(module, 'PredictionModel'):
                                logger.info(f"  ✓ solution.py imports correctly")
                    except Exception as e:
                        logger.warning(f"  ⚠️  solution.py import test failed: {e}")


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
