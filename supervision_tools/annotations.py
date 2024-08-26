import numpy as np
from supervision import BoundingBoxAnnotator, HeatMapAnnotator, LabelAnnotator, ColorAnnotator, Detections

class annotations():
    def __init__(
        self
    ):
        """
            Args:
                scene: np.ndarray - The image to be annotated.
                detections: list - The supervision detections to be annotated.
                task: str - standard: 'detect' - The task to be performed. It can be either 'detect' or 'track'.
                **kwargs - if "times" is passed, then the labels will be annotated with the time the object has been detected.
        """
        self.bounding_box_annotator = BoundingBoxAnnotator()
        self.heatmap_annotator = HeatMapAnnotator()
        self.label_annotator = LabelAnnotator()
        self.color_annotator = ColorAnnotator()
        
    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        HEATMAP: bool = True,
        BBOX: bool = True,
        LABEL: bool = True,
        COLOR: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            Scene: np.ndarray - The image to be annotated.
            Detections: sv.Detections - The supervision detections to be annotated.
        """
        
        self.detections = detections
        self.kwargs = kwargs
        
        if self.detections.tracker_id is None:
            if 'times' in self.kwargs:
                labels = [
                    f"{class_id} {(score*100):.0f}% {time:.0f}s"
                    for class_id, score, time in zip(self.detections.data['class_name'], self.detections.confidence, self.kwargs['time'])
                ]
            else:
                labels = [
                    f"{class_id} {(score*100):.0f}%"
                    for class_id, score in zip(self.detections.data['class_name'], self.detections.confidence)
                ]
        else:
            if 'times' in self.kwargs:
                labels = [
                    f"#{track_id} {class_id} {(score*100):.0f}% {time:.0f}s"
                    for track_id, class_id, score, time in zip(
                        self.detections.tracker_id,
                        self.detections.data['class_name'], 
                        self.detections.confidence,
                        self.kwargs['times']
                    )
                ]
            else:
                labels = [
                    f"#{track_id} {class_id} {(score*100):.0f}%"
                    for track_id, class_id, score in zip(
                        self.detections.tracker_id,
                        self.detections.data['class_name'], 
                        self.detections.confidence
                    )
                ]
        if 'label' in self.kwargs:
            labels = labels.append(self.kwargs['label'])
        
        
        if HEATMAP:
            scene = self.heatmap_annotator.annotate(
                scene=scene, 
                detections=self.detections
            )
        if BBOX:
            scene = self.bounding_box_annotator.annotate(
                scene=scene, 
                detections=self.detections
            )
        if LABEL:
            scene = self.label_annotator.annotate(
                scene=scene, 
                detections=self.detections, 
                labels=labels
            )
        if COLOR:
            scene = self.color_annotator.annotate(
                scene=scene, 
                detections=self.detections
            )
        scene
        
        
        

class annotations_no_labels():
    def __init__(
        self
    ):
        """
            Args:
                scene: np.ndarray - The image to be annotated.
                detections: list - The supervision detections to be annotated.
                task: str - standard: 'detect' - The task to be performed. It can be either 'detect' or 'track'.
                **kwargs - if "times" is passed, then the labels will be annotated with the time the object has been detected.
        """
        self.bounding_box_annotator = BoundingBoxAnnotator()
        self.heatmap_annotator = HeatMapAnnotator()
        self.color_annotator = ColorAnnotator()
        
    def annotate(
        self,
        scene: np.ndarray,
        detections: Detections,
        HEATMAP: bool = True,
        BBOX: bool = True,
        COLOR: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            Scene: np.ndarray - The image to be annotated.
            Detections: sv.Detections - The supervision detections to be annotated.
        """
        self.detections = detections
        self.kwargs = kwargs
        
        if HEATMAP:
            scene = self.heatmap_annotator.annotate(
                scene=scene, 
                detections=self.detections
            )
        if BBOX:
            scene = self.bounding_box_annotator.annotate(
                scene=scene, 
                detections=self.detections
            )
        if COLOR:
            scene = self.color_annotator.annotate(
                scene=scene, 
                detections=self.detections
            )
        scene