from .yolo_models import _run_detect_model, _run_pose_model

class run_yolo_model():
    """
        To run automatically YOLO models, whether it be for detection, tracking or pose estimation.
        Args:
            model_path (str): The path to the model file.
            task (str): The task to be performed by the model. It can be either 'track', 'detect' or 'pose'.
            format (str): The format of the model file. It can be 'openvino', 'onnx' or 'pt'.
            **kwargs: if "classes" is passed, then the model will filter the class detections.
    """
    def __init__(
        self,
        model: str = '',
        task: str = '',
        format: str = '',
        device: str = 'cpu',
        **kwargs
    ):
        self.task = task
        self.kwargs = kwargs
        self.device = device
        
        # Both track and detect models use supervision tracking, so we will use detect model and supress the track model
        if task == '' or task == 'detect' or task == 'track':
            self.task = 'detect'
            self.model = _run_detect_model(model_path=model, format=format, device=device, **kwargs)
        elif task == "pose":
            self.model = _run_pose_model(model_path=model, format=format, device=device, **kwargs)
    
    def __call__(
        self, 
        image_path: str
    ):
        return self.model(image_path)