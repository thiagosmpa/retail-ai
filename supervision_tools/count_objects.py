from supervision import Point, Position, LineZone, LineZoneAnnotator, Detections

class count_objects():
    def __init__(
        self,
        point_A: tuple,
        point_B: tuple,
    ):
        line_start = Point(point_A[0], point_A[1])
        line_end = Point(point_B[0], point_B[1])
        
        self.line_counter = LineZone(line_start, line_end, triggering_anchors=[Position.CENTER])
        self.annotator = LineZoneAnnotator()
        self.count_in = self.line_counter.in_count
        self.count_out = self.line_counter.out_count
    
    def trigger(
        self,
        detections: Detections
    ):
        self.line_counter.trigger(detections)
    
    def annotate(
        self,
        scene,
    ):
        self.annotator.annotate(scene, self.line_counter)
    
    def get_crossed(
        self
    ):
        return self.count_in, self.count_out