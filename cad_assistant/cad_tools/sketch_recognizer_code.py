import math
import json
import copy
import numpy as np
from sketchgraphs.data import plotting
# SketchGraphs imports
from sketchgraphs.data import Circle, Arc, Line, Point, Sketch
from sketchgraphs.data import data_utils

# Direct import of only what we need (avoiding heavy sketch_utils module)
from sketchgraphs.data.prerender_images import render_sketch


class PlottingDefaults:
    """Default values for plotting operations."""
    FONT_SIZE = 1.3
    MARKER_ANGLE_DEGREES = 45
    MARKER_PADDING = 0.12
    MARKER_LINEWIDTH = 0.05
    GRID_LINEWIDTH = 0.5

class GeometryDefaults:
    """Default values for geometry operations."""
    ROUNDING_PRECISION = 4
    TARGET_CENTER = (0.5, 0.5)
    COORDINATE_RANGE = (-0.5, 0.5)
    AXIS_RANGE = (-0.6, 0.6)
    GRID_TICKS = 11
    IMAGE_SIZE = 512
    DPI = 300


class PlottingEnhancer:
    """Enhances plotting functionality with markers and labels."""
    
    def __init__(self):
        self.original_plot_functions = {
            Arc: plotting._PLOT_BY_TYPE[Arc],
            Line: plotting._PLOT_BY_TYPE[Line],
            Point: plotting._PLOT_BY_TYPE[Point],
            Circle: plotting._PLOT_BY_TYPE[Circle]
        }
    
    def modify_plotting(self, font_size: float = None) -> None:
        """
        Modify global plotting functions to add entity markers.
        
        Args:
            font_size: Font size for markers (default: from PlottingDefaults)
        """
        if font_size is None:
            font_size = PlottingDefaults.FONT_SIZE
        
        plotting._PLOT_BY_TYPE[Arc] = self._create_arc_plotter(font_size)
        plotting._PLOT_BY_TYPE[Line] = self._create_line_plotter(font_size)
        plotting._PLOT_BY_TYPE[Point] = self._create_point_plotter(font_size)
        plotting._PLOT_BY_TYPE[Circle] = self._create_circle_plotter(font_size)
    
    def _create_line_plotter(self, font_size: float):
        """Create enhanced line plotting function."""
        def line_with_markers(*args, **kwargs):
            result = self.original_plot_functions[Line](*args, **kwargs)
            ax, line = args[0], args[1]
            self._add_text_marker(ax, line.entityId, line.mid_point, font_size)
            return result
        return line_with_markers
    
    def _create_arc_plotter(self, font_size: float):
        """Create enhanced arc plotting function."""
        def arc_with_markers(*args, **kwargs):
            result = self.original_plot_functions[Arc](*args, **kwargs)
            ax, arc = args[0], args[1]
            self._add_text_marker(ax, arc.entityId, arc.mid_point, font_size)
            return result
        return arc_with_markers
    
    def _create_point_plotter(self, font_size: float):
        """Create enhanced point plotting function."""
        def point_with_markers(*args, **kwargs):
            result = self.original_plot_functions[Point](*args, **kwargs)
            ax, point = args[0], args[1]
            self._add_text_marker(ax, point.entityId, (point.x, point.y), font_size)
            return result
        return point_with_markers
    
    def _create_circle_plotter(self, font_size: float):
        """Create enhanced circle plotting function."""
        def circle_with_markers(*args, **kwargs):
            result = self.original_plot_functions[Circle](*args, **kwargs)
            ax, circle = args[0], args[1]
            
            # Calculate marker position on circumference
            center = (circle.xCenter, circle.yCenter)
            angle_radians = math.radians(PlottingDefaults.MARKER_ANGLE_DEGREES)
            marker_x = center[0] + circle.radius * math.cos(angle_radians)
            marker_y = center[1] + circle.radius * math.sin(angle_radians)
            
            self._add_text_marker(ax, circle.entityId, (marker_x, marker_y), font_size)
            return result
        return circle_with_markers
    
    @staticmethod
    def _add_text_marker(ax, entity_id: str, position: Tuple[float, float], font_size: float) -> None:
        """Add a text marker to the plot."""
        ax.text(
            position[0], position[1], str(entity_id),
            fontsize=font_size,
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                boxstyle=f'round,pad={PlottingDefaults.MARKER_PADDING}',
                linewidth=PlottingDefaults.MARKER_LINEWIDTH
            ),
            ha='center', va='center'
        )


# =============================================================================
# Core Conversion Classes
# =============================================================================

class SketchConverter:
    """Handles conversion between FreeCAD sketches and various formats."""
    
    @staticmethod
    def freecad_to_json(sketch: Any, is_construction_dict: Optional[Dict[str, bool]] = None) -> str:
        """
        Convert a FreeCAD sketch to JSON format.
        
        Args:
            sketch: FreeCAD sketch object
            is_construction_dict: Optional mapping of geometry indices to construction status
            
        Returns:
            JSON string representation of the sketch
        """
        geometry_data = []
        constraint_data = []
        
        # Process geometry
        for index, geometry in enumerate(sketch.Geometry):
            is_construction = (
                is_construction_dict.get(str(index), False) 
                if is_construction_dict else False
            )
            
            geometry_dict = SketchConverter._convert_geometry_to_dict(
                geometry, index, is_construction
            )
            geometry_data.append(geometry_dict)
        
        # Process constraints
        for index, constraint in enumerate(sketch.Constraints):
            constraint_dict = SketchConverter._convert_constraint_to_dict(constraint, index)
            constraint_data.append(constraint_dict)
        
        # Create final JSON structure
        json_data = {
            "Geometry": SketchConverter._remove_none_values(geometry_data),
            "Constraints": SketchConverter._remove_none_values(constraint_data)
        }
        
        return json.dumps(json_data, indent=4)
    
    @staticmethod
    def _convert_geometry_to_dict(geometry: Any, index: int, is_construction: bool) -> Dict[str, Any]:
        """Convert a single geometry element to dictionary format."""
        base_dict = {
            "Id": index,
            "isConstruction": is_construction
        }
        
        if geometry.TypeId == 'Part::GeomLineSegment':
            return {
                **base_dict,
                "Type": "Line segment",
                "X": None,
                "Y": None,
                "StartPoint": {
                    "x": round(geometry.StartPoint.x, GeometryDefaults.ROUNDING_PRECISION),
                    "y": round(geometry.StartPoint.y, GeometryDefaults.ROUNDING_PRECISION)
                },
                "EndPoint": {
                    "x": round(geometry.EndPoint.x, GeometryDefaults.ROUNDING_PRECISION),
                    "y": round(geometry.EndPoint.y, GeometryDefaults.ROUNDING_PRECISION)
                }
            }
        
        elif geometry.TypeId == 'Part::GeomArcOfCircle':
            return {
                **base_dict,
                "Type": "Arc",
                "X": None,
                "Y": None,
                "StartPoint": {
                    "x": round(geometry.StartPoint.x, GeometryDefaults.ROUNDING_PRECISION),
                    "y": round(geometry.StartPoint.y, GeometryDefaults.ROUNDING_PRECISION)
                },
                "EndPoint": {
                    "x": round(geometry.EndPoint.x, GeometryDefaults.ROUNDING_PRECISION),
                    "y": round(geometry.EndPoint.y, GeometryDefaults.ROUNDING_PRECISION)
                },
                "Center": {
                    "x": round(geometry.Center.x, GeometryDefaults.ROUNDING_PRECISION),
                    "y": round(geometry.Center.y, GeometryDefaults.ROUNDING_PRECISION)
                },
                "Radius": round(geometry.Radius, GeometryDefaults.ROUNDING_PRECISION),
                "FirstParameter": round(geometry.FirstParameter, GeometryDefaults.ROUNDING_PRECISION),
                "LastParameter": round(geometry.LastParameter, GeometryDefaults.ROUNDING_PRECISION)
            }
        
        elif geometry.TypeId == 'Part::GeomCircle':
            return {
                **base_dict,
                "Type": "Circle",
                "Center": {
                    "x": round(geometry.Center.x, GeometryDefaults.ROUNDING_PRECISION),
                    "y": round(geometry.Center.y, GeometryDefaults.ROUNDING_PRECISION)
                },
                "Radius": round(geometry.Radius, GeometryDefaults.ROUNDING_PRECISION)
            }
        
        elif geometry.TypeId == 'Part::GeomPoint':
            return {
                **base_dict,
                "Type": "Point",
                "X": round(geometry.X, GeometryDefaults.ROUNDING_PRECISION),
                "Y": round(geometry.Y, GeometryDefaults.ROUNDING_PRECISION)
            }
        
        return base_dict
    
    @staticmethod
    def _convert_constraint_to_dict(constraint: Any, index: int) -> Dict[str, Any]:
        """Convert a single constraint to dictionary format."""
        ref_first = constraint.First
        ref_first_pos = (
            PointOfGeometry(constraint.FirstPos).name 
            if constraint.FirstPos > 0 else None
        )
        ref_second = constraint.Second if constraint.Second > 0 else None
        ref_second_pos = (
            PointOfGeometry(constraint.SecondPos).name 
            if constraint.Second > 0 and constraint.SecondPos > 0 else None
        )
        
        return {
            "Id": index,
            "Type": constraint.Type,
            "First": ref_first,
            "FirstPos": ref_first_pos,
            "Second": ref_second,
            "SecondPos": ref_second_pos
        }
    
    @staticmethod
    def _remove_none_values(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove None values from dictionaries in a list."""
        return [{k: v for k, v in d.items() if v is not None} for d in data_list]
    
    @staticmethod
    def freecad_to_sketchgraphs(sketch: Any, is_construction_dict: Optional[Dict[str, bool]] = None) -> Sketch:
        """
        Convert a FreeCAD sketch to SketchGraphs format.
        
        Args:
            sketch: FreeCAD sketch object
            is_construction_dict: Optional mapping of geometry indices to construction status
            
        Returns:
            SketchGraphs Sketch object
        """
        sketch_onshape = Sketch()
        
        for geometry_id, geometry in enumerate(sketch.Geometry):
            entity_id = str(geometry_id)
            is_construction = (
                is_construction_dict.get(entity_id, False) 
                if is_construction_dict else False
            )
            
            entity = SketchConverter._convert_geometry_to_entity(
                geometry, entity_id, is_construction
            )
            
            if entity:
                sketch_onshape.entities[entity_id] = entity
        
        return sketch_onshape
    
    @staticmethod
    def _convert_geometry_to_entity(geometry: Any, entity_id: str, is_construction: bool) -> Optional[Any]:
        """Convert FreeCAD geometry to SketchGraphs entity."""
        if geometry.TypeId == 'Part::GeomLineSegment':
            line = Line.from_points(
                Point(entityId=None, x=geometry.StartPoint[0], y=geometry.StartPoint[1]),
                Point(entityId=None, x=geometry.EndPoint[0], y=geometry.EndPoint[1])
            )
            line.entityId = entity_id
            line.isConstruction = is_construction
            return line
        
        elif geometry.TypeId == 'Part::GeomPoint':
            point = Point(entityId=entity_id, x=geometry.X, y=geometry.Y)
            point.isConstruction = is_construction
            return point
        
        elif geometry.TypeId == 'Part::GeomCircle':
            circle = Circle(
                entity_id, False, geometry.Center.x, geometry.Center.y, 
                radius=geometry.Radius
            )
            circle.isConstruction = is_construction
            return circle
        
        elif geometry.TypeId == 'Part::GeomArcOfCircle':
            clockwise = geometry.Axis[2] == -1
            arc = Arc.from_info({
                "id": entity_id,
                "center": [geometry.Center[0], geometry.Center[1]],
                "startPoint": [geometry.StartPoint[0], geometry.StartPoint[1]],
                "endPoint": [geometry.EndPoint[0], geometry.EndPoint[1]],
                "radius": geometry.Radius,
                "clockwise": clockwise,
            })
            arc.isConstruction = is_construction
            return arc
        
        return None


# =============================================================================
# Sketch Processing Utilities
# =============================================================================

class SketchProcessor:
    """Utilities for processing and normalizing sketches."""
    
    @staticmethod
    def center_sketch(sketch: Sketch, target_center: Tuple[float, float] = None) -> Sketch:
        """
        Center a sketch around a target point.
        
        Args:
            sketch: Input sketch to center
            target_center: Target center coordinates (default: (0.5, 0.5))
            
        Returns:
            New centered sketch
        """
        if target_center is None:
            target_center = GeometryDefaults.TARGET_CENTER
        
        new_sketch = copy.deepcopy(sketch)
        data_utils.normalize_sketch(new_sketch)
        
        # Get bounding box
        (x0, y0), (x1, y1) = data_utils.get_sketch_bbox(new_sketch)
        
        # Calculate shifts needed
        current_center_x = np.mean([x0, x1])
        current_center_y = np.mean([y0, y1])
        x_shift = target_center[0] - current_center_x
        y_shift = target_center[1] - current_center_y
        
        # Apply shifts to entities
        for entity in new_sketch.entities.values():
            SketchProcessor._apply_shift_to_entity(entity, x_shift, y_shift)
        
        return new_sketch
    
    @staticmethod
    def _apply_shift_to_entity(entity: Any, x_shift: float, y_shift: float) -> None:
        """Apply coordinate shifts to a sketch entity."""
        pos_params = data_utils.POS_PARAMS.get(entity.type)
        if pos_params is None:
            return
        
        for param_id in pos_params:
            shift_value = x_shift if "x" in param_id.lower() else y_shift
            current_value = getattr(entity, param_id)
            setattr(entity, param_id, current_value + shift_value)



# =============================================================================
# Main Interface Class
# =============================================================================

class SketchRecognizer:
    """Main interface for sketch recognition and processing."""
    
    def __init__(self):
        self.plotting_enhancer = PlottingEnhancer()
        self.plotting_enhancer.modify_plotting()
    
    def process_sketch(self, sketch: Any, return_image: bool = True, 
                      is_construction_dict: Optional[Dict[str, bool]] = None,
                      logdir: str = None, step: int = None):
        """
        Process a FreeCAD sketch and return JSON and/or image representation.
        
        Args:
            sketch: FreeCAD sketch object
            return_image: Whether to return rendered image
            is_construction_dict: Optional construction status mapping
            logdir: Directory for logging intermediate results
            step: Current step number for logging
            
        Returns:
            JSON string if return_image=False, PIL Image if return_image=True
        """
        # Generate JSON representation
        serialized_sketch = SketchConverter.freecad_to_json(sketch, is_construction_dict)
        
        if not return_image:
            return serialized_sketch
        
        # Convert to SketchGraphs format and render
        onshape_sketch = SketchConverter.freecad_to_sketchgraphs(sketch, is_construction_dict)
        data_utils.normalize_sketch(onshape_sketch)
        
        sketch_img = render_sketch(
            onshape_sketch,
            size_pixels=GeometryDefaults.IMAGE_SIZE,
            return_img=True,
            is_colored=False,
            linewidth=1,
            pad=0.1,
            sketch_extent=1,
            show_borders=False
        )
        
        print("The sketch contains the following geometries and constraints, serialized in JSON format:")
        print(serialized_sketch)
        print("Rendered image of the sketch:")
        
        # Optional logging to files
        if logdir and step is not None:
            # Save image
            img_filename = os.path.join(logdir, f"sketch_step_{step}.png")
            sketch_img.save(img_filename)
            
        return sketch_img

def sketch_recognizer(sketch: Any, returnImage: bool = True, logdir: str = None, step: int = None, is_construction_dict = None) -> Any:
    """
    Main interface for LLM agents to process FreeCAD sketches.
    
    Args:
        sketch: FreeCAD sketch object
        returnImage: Whether to return image (True) or JSON (False)
        logdir: Directory for logging (defaults to global logdir)
        step: Current step number (defaults to global step)
        
    Returns:
        PIL Image or JSON string depending on returnImage parameter
    """
    # Use global values as defaults if not provided
    if logdir is None:
        logdir = globals().get('logdir', '')
    if step is None:
        step = globals().get('step', 0)
    if is_construction_dict is None:
        is_construction_dict = globals().get('is_construction_dict', 0)
    
    recognizer = SketchRecognizer()
    return recognizer.process_sketch(sketch, return_image=returnImage, logdir=logdir, step=step, is_construction_dict=is_construction_dict)