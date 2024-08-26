from supervision import Detections

def _update_tracker(
    detections,
    tracker
):
    return tracker.update_with_detections(detections=detections)

def get_ultralytics_detections(
    results,
    tracker = None
):
    detections = Detections.from_ultralytics(results)
    
    if tracker is not None:
        detections = _update_tracker(detections, tracker)
    
    return detections
