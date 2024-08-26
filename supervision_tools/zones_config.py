import numpy as np
import json
from typing import List
from supervision import (
    PolygonZone,
    Position
)

def load_zones_config(file_path: str) -> List[np.ndarray]:
    """
    Load polygon zone configurations from a JSON file.

    This function reads a JSON file which contains polygon coordinates, and
    converts them into a list of NumPy arrays. Each polygon is represented as
    a NumPy array of coordinates.

    Args:
        file_path (str): The path to the JSON configuration file.

    Returns:
        List[np.ndarray]: A list of polygons, each represented as a NumPy array.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data]

def create_zones(
    polygons,
    video_width,
    video_height
):
    resolution_wh = (video_width, video_height)
    zones = [
        PolygonZone(
            polygon=polygon,
            frame_resolution_wh=resolution_wh,
            triggering_anchors=(Position.CENTER)
        )
        for polygon in polygons
    ]
    return zones
    
    