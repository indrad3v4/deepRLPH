# -*- coding: utf-8 -*-
"""
dataset_inspector.py - Dataset Schema Detection & Validation (BE-002)

Automatically extracts schema from datasets (Parquet, CSV, NumPy) to enable
data-aware PRD generation and validation before training.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger("DatasetInspector")


@dataclass
class DatasetSchema:
    """Schema information extracted from dataset"""
    
    file_path: str
    file_type: str  # parquet, csv, numpy
    columns: List[str]
    dtypes: Dict[str, str]
    shape: tuple  # (rows, cols) or (rows, timesteps, features)
    detected_type: str  # time_series, tabular, image
    num_features: Optional[int] = None
    sequence_length: Optional[int] = None
    num_samples: Optional[int] = None
    memory_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetSchema":
        return cls(**data)


class DatasetInspector:
    """Extract schema from various dataset formats"""
    
    def __init__(self):
        self.supported_formats = ['.parquet', '.csv', '.npy', '.npz']
    
    def inspect(self, file_path: Union[str, Path]) -> DatasetSchema:
        """
        Inspect dataset file and extract schema.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            DatasetSchema with extracted information
            
        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: {self.supported_formats}"
            )
        
        logger.info(f"📊 Inspecting dataset: {file_path.name}")
        
        if suffix == '.parquet':
            return self._inspect_parquet(file_path)
        elif suffix == '.csv':
            return self._inspect_csv(file_path)
        elif suffix in ['.npy', '.npz']:
            return self._inspect_numpy(file_path)
        else:
            raise ValueError(f"Handler not implemented for {suffix}")
    
    def _inspect_parquet(self, file_path: Path) -> DatasetSchema:
        """Inspect Parquet file (requires pandas + pyarrow)"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for Parquet inspection. Install: pip install pandas pyarrow")
        
        df = pd.read_parquet(file_path)
        
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        shape = df.shape
        
        # Detect if time-series (heuristic: many numeric columns, sequence-like structure)
        detected_type = self._detect_type(df, columns)
        
        # Extract time-series specific info if applicable
        num_features = None
        sequence_length = None
        num_samples = shape[0]
        
        if detected_type == "time_series":
            # Assume columns are features (adjust based on actual structure)
            num_features = len([c for c in columns if c not in ['id', 'target', 'timestamp']])
            
            # Check if data has sequence structure (e.g., flattened sequences)
            # For Wundernn-style: each row might be a sequence
            if len(df) > 0 and hasattr(df.iloc[0, 0], '__len__'):
                # First column might contain sequences
                sequence_length = len(df.iloc[0, 0]) if hasattr(df.iloc[0, 0], '__len__') else None
            else:
                # Flat structure: infer from metadata or defaults
                sequence_length = None
        
        memory_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        logger.info(f"  Shape: {shape}, Type: {detected_type}, Memory: {memory_size_mb:.2f} MB")
        
        return DatasetSchema(
            file_path=str(file_path),
            file_type="parquet",
            columns=columns,
            dtypes=dtypes,
            shape=shape,
            detected_type=detected_type,
            num_features=num_features,
            sequence_length=sequence_length,
            num_samples=num_samples,
            memory_size_mb=memory_size_mb,
        )
    
    def _inspect_csv(self, file_path: Path) -> DatasetSchema:
        """Inspect CSV file"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for CSV inspection. Install: pip install pandas")
        
        # Read first 1000 rows for quick inspection
        df = pd.read_csv(file_path, nrows=1000)
        
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Get full row count without loading entire file
        with open(file_path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1  # subtract header
        
        shape = (total_rows, len(columns))
        detected_type = self._detect_type(df, columns)
        
        memory_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        logger.info(f"  Shape: {shape}, Type: {detected_type}, Size: {memory_size_mb:.2f} MB")
        
        return DatasetSchema(
            file_path=str(file_path),
            file_type="csv",
            columns=columns,
            dtypes=dtypes,
            shape=shape,
            detected_type=detected_type,
            num_features=len(columns),
            num_samples=total_rows,
            memory_size_mb=memory_size_mb,
        )
    
    def _inspect_numpy(self, file_path: Path) -> DatasetSchema:
        """Inspect NumPy array file"""
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy required for .npy/.npz inspection")
        
        if file_path.suffix == '.npy':
            arr = np.load(file_path)
            shape = arr.shape
            dtype = str(arr.dtype)
            
            # Infer structure: (samples, timesteps, features) or (samples, features)
            detected_type = "time_series" if len(shape) == 3 else "tabular"
            
            num_samples = shape[0]
            sequence_length = shape[1] if len(shape) >= 2 else None
            num_features = shape[2] if len(shape) == 3 else (shape[1] if len(shape) == 2 else None)
            
            memory_size_mb = arr.nbytes / (1024 * 1024)
            
            return DatasetSchema(
                file_path=str(file_path),
                file_type="numpy",
                columns=[f"feature_{i}" for i in range(num_features or shape[-1])],
                dtypes={"all": dtype},
                shape=shape,
                detected_type=detected_type,
                num_features=num_features,
                sequence_length=sequence_length,
                num_samples=num_samples,
                memory_size_mb=memory_size_mb,
            )
        
        else:  # .npz
            data = np.load(file_path)
            arrays = {key: data[key] for key in data.files}
            
            # Take first array as primary
            primary_key = data.files[0]
            primary_arr = arrays[primary_key]
            
            shape = primary_arr.shape
            detected_type = "time_series" if len(shape) == 3 else "tabular"
            
            return DatasetSchema(
                file_path=str(file_path),
                file_type="numpy",
                columns=data.files,
                dtypes={k: str(v.dtype) for k, v in arrays.items()},
                shape=shape,
                detected_type=detected_type,
                num_samples=shape[0],
                memory_size_mb=sum(arr.nbytes for arr in arrays.values()) / (1024 * 1024),
            )
    
    def _detect_type(self, df, columns: List[str]) -> str:
        """
        Heuristic to detect dataset type.
        
        Rules:
        - If mostly numeric columns + high column count → time_series
        - If image-like dimensions → image
        - Else → tabular
        """
        import pandas as pd
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        numeric_ratio = len(numeric_cols) / len(columns)
        
        # Heuristic: >80% numeric columns + many features → time-series
        if numeric_ratio > 0.8 and len(columns) > 10:
            return "time_series"
        
        # Check for image-like patterns (height, width, channels)
        if len(df.shape) == 3 or any(dim in [28, 32, 64, 224, 256] for dim in df.shape):
            return "image"
        
        return "tabular"
    
    def inspect_directory(self, directory: Union[str, Path]) -> Dict[str, DatasetSchema]:
        """
        Inspect all datasets in a directory.
        
        Returns:
            Dict mapping filename to schema
        """
        directory = Path(directory)
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        schemas = {}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    schema = self.inspect(file_path)
                    schemas[file_path.name] = schema
                except Exception as e:
                    logger.warning(f"Failed to inspect {file_path.name}: {e}")
        
        return schemas


def main():
    """CLI for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset_inspector.py <file_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    inspector = DatasetInspector()
    
    if path.is_file():
        schema = inspector.inspect(path)
        print(schema.to_json())
    elif path.is_dir():
        schemas = inspector.inspect_directory(path)
        print(json.dumps({name: schema.to_dict() for name, schema in schemas.items()}, indent=2))
    else:
        print(f"Error: {path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
