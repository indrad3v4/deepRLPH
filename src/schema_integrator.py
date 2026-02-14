# -*- coding: utf-8 -*-
"""
schema_integrator.py - BE-002.2: Schema Integration Helper

Integrates DatasetInspector with orchestrator to auto-detect and attach
schema metadata to projects.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

from dataset_inspector import DatasetInspector, DatasetSchema

logger = logging.getLogger("SchemaIntegrator")


class SchemaIntegrator:
    """Integrates dataset schema detection into project workflow"""
    
    def __init__(self):
        self.inspector = DatasetInspector()
    
    def inspect_and_save(
        self,
        project_dir: Path,
        dataset_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Inspect datasets in project and save schema.
        
        Args:
            project_dir: Project root directory
            dataset_path: Specific dataset file (or None to scan data/raw/)
            
        Returns:
            Dict with schemas for all datasets found
        """
        try:
            schemas = {}
            
            if dataset_path and dataset_path.exists():
                # Inspect single file
                schema = self.inspector.inspect(dataset_path)
                schemas[dataset_path.name] = schema.to_dict()
            else:
                # Scan data/raw/ directory
                data_dir = project_dir / "data" / "raw"
                if data_dir.exists():
                    found_schemas = self.inspector.inspect_directory(data_dir)
                    schemas = {name: schema.to_dict() for name, schema in found_schemas.items()}
            
            if not schemas:
                logger.warning("No datasets found to inspect")
                return {"status": "no_datasets", "schemas": {}}
            
            # Save to project
            schema_file = project_dir / "dataset_schema.json"
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(schemas, f, indent=2)
            
            logger.info(f"✅ Schema saved: {schema_file} ({len(schemas)} datasets)")
            
            return {
                "status": "success",
                "schemas": schemas,
                "schema_file": str(schema_file)
            }
            
        except Exception as e:
            logger.error(f"Schema inspection failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def load_schema(self, project_dir: Path) -> Optional[Dict[str, Any]]:
        """Load previously saved schema from project"""
        schema_file = project_dir / "dataset_schema.json"
        
        if not schema_file.exists():
            return None
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return None
    
    def attach_to_metadata(
        self,
        config_metadata: Dict[str, Any],
        schemas: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Attach schemas to project metadata for PRD generation.
        
        Returns:
            Updated metadata dict
        """
        metadata = dict(config_metadata or {})
        metadata['dataset_schemas'] = schemas
        
        # Extract aggregate stats for easy access
        if schemas:
            first_schema = list(schemas.values())[0]
            metadata['num_features'] = first_schema.get('num_features')
            metadata['sequence_length'] = first_schema.get('sequence_length')
            metadata['num_samples'] = first_schema.get('num_samples')
            metadata['detected_type'] = first_schema.get('detected_type')
        
        return metadata
