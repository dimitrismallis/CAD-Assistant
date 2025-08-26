"""
FreeCAD Integration Module

This module provides utilities for working with FreeCAD sketches, including:
- Converting FreeCAD sketches to JSON format
- Converting between FreeCAD and SketchGraphs formats
- Rendering sketches with visual enhancements
- Sketch normalization and processing

NOTE: The print statements are important as they inform the LLM about available imports.
"""

import sys
import os

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

print("import sys, os")
import json
# FreeCAD imports
sys.path.insert(0, '/usr/lib/freecad/lib')
import FreeCAD as App
import Part, Sketcher
print("import FreeCAD as App")
print("import Part, Sketcher")

# Scientific computing

print("import numpy as np")
print("import re, math")


class PointOfGeometry(Enum):
    """Enumeration for geometry point references."""
    START = 1
    END = 2
    CENTER = 3


# =============================================================================
# Initialization and Setup
# =============================================================================

def initialize_freecad_environment(project_file: str = None) -> Tuple[Any, Any]:
    """
    Initialize FreeCAD environment and load or create a document.
    
    Args:
        project_file: Optional path to existing FreeCAD project file
        
    Returns:
        Tuple of (document, sketch) objects
    """

    if project_file and os.path.exists(project_file):
        # Load existing project
        doc = App.openDocument(project_file)
        print(f"doc = App.openDocument('project.FCStd')")
        if len(doc.Objects) == 1:
            if doc.Objects[0].Name == 'sketch':
                sketch = doc.Objects[0]
                print("sketch = doc.Objects[0]")


    else:
        # Create new document
        doc = App.newDocument('sketch')
        sketch = doc.addObject('Sketcher::SketchObject', 'sketch')
        print("doc = App.newDocument('sketch')")
        print("sketch = doc.addObject('Sketcher::SketchObject', 'sketch')")
    
    return doc, doc.Objects[0]