import os
import sys
from PIL import Image, ImageDraw, ImageFont
import FreeCAD as App
import Part
import subprocess
import json
from typing import Dict, List, Any, Optional, Tuple
from cad_assistant.cad_tools.sketch_recognizer_code import SketchRecognizer

def solid_recognizer(doc: App.Document, logdir: str = None, step: int = None) -> None:
    """Analyzes a 3D CAD Model, providing both a JSON serialization of the Attributes for the geometry 
    and constraints and an image rendering of the sketch and extrude opeartions. Use this function to understand the current FreeCAD sketch.

    This function processes the given sketch and returns a JSON representation describing its sketches and extrusions. 
    Attributes for Geometries and constraints are extracted directly from these lists, 
    serialized into a structured JSON format for easy interpretation. 
    
    Additionally, this function generates an image rendering of the 3D CAD model for visual examination. The sketch
    is with multiple views.
    
    Parameters:
    ----------
    doc (App.Document): 
        The FreeCAD document including a list of Objects that can be sketch and extusion operations
            
    Prints:
    -------
    Displays the sketch and extrusion parameters for quick review. It also returns a
    cad_image (PIL.Image.Image or np.ndarray) rendering of the sketch."""


    # Use global values as defaults if not provided
    if logdir is None:
        logdir = globals().get('logdir', '')
    if step is None:
        step = globals().get('step', 0)

    json_description = {}
    for obj in doc.Objects:
        if obj.TypeId == 'Sketcher::SketchObject': 
            recognizer = SketchRecognizer()
            json_description[obj.Name] = json.loads(recognizer.process_sketch(obj, return_image=False, logdir=logdir, step=step))
        elif obj.TypeId == 'Part::Extrusion':
            temp_dic = {}
            temp_dic['Base'] = obj.Base.Name
            temp_dic['Dir'] =  {"x": round(obj.Dir.x, 4), "y": round(obj.Dir.y, 4), "z": round(obj.Dir.z, 4)},
            temp_dic['LengthFwd'] = str(obj.LengthFwd)
            temp_dic['LengthRev'] = str(obj.LengthRev)
            temp_dic['Symmetric'] = obj.Symmetric
            temp_dic['Reversed'] = obj.Reversed
            temp_dic['Reversed'] = obj.Reversed    
            json_description[obj.Name] = temp_dic
    print("The 3D CAD model contains the following sketch and extrusion operations, serialized in JSON format:")
    serialization = json.dumps(json_description, indent=4)
    print(serialization)
    
    print("Rendered image of the 3D Model:")


    img_filename = os.path.join(logdir, f"solid_step_{step}.png")
    step_filename = os.path.join(logdir, f"solid_step_{step}.stp")

    try:
        part = [f for f in doc.Objects if f.Module=='Part'][0]
        
        Part.export([part],step_filename)
    except:
        print("Solid was not detected.")

    subprocess.run(["xvfb-run", '/opt/conda/envs/occenv/bin/python', "cad_assistant/cad_tools/solid_recognizer_occ.py",step_filename])
    if not os.path.exists(img_filename):
        print("Solid rendering could not be generated.")

    img = Image.open(img_filename)
    os.rename(img_filename, os.path.join(logdir, f"step_image_{step}.png"))


    return img
