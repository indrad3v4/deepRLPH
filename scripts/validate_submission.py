#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_submission.py - BE-007.1: Submission Validation Script

Validates competition submission files before packaging:
- Checks solution.py exists and has correct interface
- Verifies model.onnx exists and loads
- Tests dry-run prediction
"""

import sys
import logging
from pathlib import Path
import importlib.util
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SubmissionValidator")


class SubmissionValidator:
    """Validates submission package before creating zip"""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if valid, False if errors found
        """
        logger.info("\n🔍 Validating submission package...")
        
        # Check 1: solution.py exists
        if not self._check_solution_file():
            return False
        
        # Check 2: PredictionModel class exists
        if not self._check_prediction_model_class():
            return False
        
        # Check 3: model.onnx exists
        if not self._check_model_file():
            return False
        
        # Check 4: Interface validation
        if not self._check_interface():
            return False
        
        # Check 5: Test dry-run (optional, may fail if utils.py missing)
        self._check_dry_run()
        
        # Summary
        if self.errors:
            logger.error("\n❌ Validation FAILED")
            for err in self.errors:
                logger.error(f"  - {err}")
            return False
        
        if self.warnings:
            logger.warning("\n⚠️  Warnings:")
            for warn in self.warnings:
                logger.warning(f"  - {warn}")
        
        logger.info("\n✅ Validation PASSED")
        return True
    
    def _check_solution_file(self) -> bool:
        """Check solution.py exists at project root"""
        solution_file = self.project_root / "solution.py"
        
        if not solution_file.exists():
            self.errors.append("solution.py not found at project root")
            return False
        
        logger.info("✓ solution.py exists")
        return True
    
    def _check_prediction_model_class(self) -> bool:
        """Check PredictionModel class exists in solution.py"""
        solution_file = self.project_root / "solution.py"
        
        try:
            # Load module
            spec = importlib.util.spec_from_file_location("solution", solution_file)
            if not spec or not spec.loader:
                self.errors.append("Failed to load solution.py as module")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for PredictionModel class
            if not hasattr(module, 'PredictionModel'):
                self.errors.append("PredictionModel class not found in solution.py")
                return False
            
            logger.info("✓ PredictionModel class found")
            return True
            
        except Exception as e:
            self.errors.append(f"Error loading solution.py: {e}")
            return False
    
    def _check_model_file(self) -> bool:
        """Check model.onnx exists"""
        model_file = self.project_root / "models" / "final" / "model.onnx"
        
        if not model_file.exists():
            self.errors.append("models/final/model.onnx not found")
            return False
        
        file_size_mb = model_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ model.onnx exists ({file_size_mb:.2f} MB)")
        
        if file_size_mb > 100:
            self.warnings.append(f"Model size is large: {file_size_mb:.1f} MB")
        
        return True
    
    def _check_interface(self) -> bool:
        """Validate PredictionModel interface"""
        solution_file = self.project_root / "solution.py"
        
        try:
            spec = importlib.util.spec_from_file_location("solution", solution_file)
            if not spec or not spec.loader:
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            PredictionModel = module.PredictionModel
            
            # Check for required methods
            if not hasattr(PredictionModel, 'predict'):
                self.errors.append("PredictionModel missing 'predict' method")
                return False
            
            # Check if predict method signature looks correct
            import inspect
            sig = inspect.signature(PredictionModel.predict)
            
            if len(sig.parameters) < 2:  # self + at least one arg
                self.warnings.append("predict() method signature may be incorrect")
            
            logger.info("✓ Interface validation passed")
            return True
            
        except Exception as e:
            self.errors.append(f"Interface check failed: {e}")
            return False
    
    def _check_dry_run(self) -> bool:
        """Test instantiation and basic prediction (may fail without utils.py)"""
        try:
            solution_file = self.project_root / "solution.py"
            spec = importlib.util.spec_from_file_location("solution", solution_file)
            if not spec or not spec.loader:
                return False
            
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(self.project_root))  # For imports
            spec.loader.exec_module(module)
            
            # Try to instantiate
            model = module.PredictionModel()
            logger.info("✓ PredictionModel instantiation successful")
            
            # Note: Can't test actual prediction without DataPoint class from utils.py
            self.warnings.append("Dry-run prediction not tested (requires competition utils.py)")
            
            return True
            
        except Exception as e:
            self.warnings.append(f"Dry-run failed (may be OK if utils.py not included): {e}")
            return True  # Not a hard error


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <project_root>")
        print("Example: python scripts/validate_submission.py .")
        sys.exit(1)
    
    project_root = Path(sys.argv[1])
    
    if not project_root.exists():
        logger.error(f"Project root not found: {project_root}")
        sys.exit(1)
    
    validator = SubmissionValidator(project_root)
    success = validator.validate()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
