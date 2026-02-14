# -*- coding: utf-8 -*-
"""
test_dataset_inspector.py - Unit tests for DatasetInspector (BE-002)
"""

import pytest
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

from src.dataset_inspector import DatasetInspector, DatasetSchema


class TestDatasetInspector:
    """Test suite for DatasetInspector"""
    
    @pytest.fixture
    def inspector(self):
        return DatasetInspector()
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_parquet_inspection(self, inspector, temp_dir):
        """Test Parquet file schema extraction"""
        # Create sample Parquet file
        df = pd.DataFrame({
            'feat_0': np.random.randn(100),
            'feat_1': np.random.randn(100),
            'feat_2': np.random.randn(100),
            'target': np.random.randn(100),
        })
        
        parquet_file = temp_dir / "test.parquet"
        df.to_parquet(parquet_file)
        
        # Inspect
        schema = inspector.inspect(parquet_file)
        
        assert schema.file_type == "parquet"
        assert len(schema.columns) == 4
        assert 'feat_0' in schema.columns
        assert schema.shape == (100, 4)
        assert schema.num_samples == 100
        assert schema.memory_size_mb > 0
    
    def test_csv_inspection(self, inspector, temp_dir):
        """Test CSV file schema extraction"""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['a', 'b', 'c', 'd', 'e'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5],
        })
        
        csv_file = temp_dir / "test.csv"
        df.to_csv(csv_file, index=False)
        
        schema = inspector.inspect(csv_file)
        
        assert schema.file_type == "csv"
        assert len(schema.columns) == 3
        assert schema.shape == (5, 3)
        assert schema.num_features == 3
    
    def test_numpy_inspection(self, inspector, temp_dir):
        """Test NumPy array schema extraction"""
        # Time-series shaped: (samples, timesteps, features)
        arr = np.random.randn(50, 100, 32)
        
        npy_file = temp_dir / "test.npy"
        np.save(npy_file, arr)
        
        schema = inspector.inspect(npy_file)
        
        assert schema.file_type == "numpy"
        assert schema.shape == (50, 100, 32)
        assert schema.detected_type == "time_series"
        assert schema.num_samples == 50
        assert schema.sequence_length == 100
        assert schema.num_features == 32
    
    def test_tabular_numpy(self, inspector, temp_dir):
        """Test tabular (2D) NumPy array"""
        arr = np.random.randn(200, 15)
        
        npy_file = temp_dir / "test.npy"
        np.save(npy_file, arr)
        
        schema = inspector.inspect(npy_file)
        
        assert schema.detected_type == "tabular"
        assert schema.shape == (200, 15)
        assert schema.num_samples == 200
    
    def test_unsupported_format(self, inspector, temp_dir):
        """Test error on unsupported file format"""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("some data")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            inspector.inspect(txt_file)
    
    def test_file_not_found(self, inspector):
        """Test error when file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            inspector.inspect("nonexistent.parquet")
    
    def test_schema_to_dict(self):
        """Test DatasetSchema serialization"""
        schema = DatasetSchema(
            file_path="test.parquet",
            file_type="parquet",
            columns=['a', 'b', 'c'],
            dtypes={'a': 'int64', 'b': 'float64', 'c': 'object'},
            shape=(100, 3),
            detected_type="tabular",
            num_features=3,
            num_samples=100,
        )
        
        schema_dict = schema.to_dict()
        
        assert schema_dict['file_path'] == "test.parquet"
        assert schema_dict['num_features'] == 3
        assert len(schema_dict['columns']) == 3
    
    def test_schema_to_json(self):
        """Test JSON serialization"""
        schema = DatasetSchema(
            file_path="test.csv",
            file_type="csv",
            columns=['x', 'y'],
            dtypes={'x': 'int', 'y': 'float'},
            shape=(50, 2),
            detected_type="tabular",
        )
        
        json_str = schema.to_json()
        parsed = json.loads(json_str)
        
        assert parsed['file_type'] == 'csv'
        assert 'columns' in parsed
    
    def test_inspect_directory(self, inspector, temp_dir):
        """Test inspecting multiple files in directory"""
        # Create multiple files
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'b': [4, 5, 6]})
        
        (temp_dir / "file1.parquet").write_bytes(df1.to_parquet())
        (temp_dir / "file2.csv").write_text(df2.to_csv(index=False))
        
        schemas = inspector.inspect_directory(temp_dir)
        
        assert len(schemas) >= 1  # At least one file should be inspected
        assert all(isinstance(s, DatasetSchema) for s in schemas.values())
    
    def test_time_series_detection(self, inspector, temp_dir):
        """Test time-series type detection heuristic"""
        # Create dataset with many numeric features (time-series pattern)
        df = pd.DataFrame({
            f'feat_{i}': np.random.randn(100) for i in range(32)
        })
        
        parquet_file = temp_dir / "timeseries.parquet"
        df.to_parquet(parquet_file)
        
        schema = inspector.inspect(parquet_file)
        
        assert schema.detected_type == "time_series"
        assert schema.num_features == 32
