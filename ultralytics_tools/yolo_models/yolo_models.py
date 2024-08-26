from ultralytics import YOLO
from ultralytics.engine.results import Results
import os
import torch

def _create_model(
    dir: str,
    format: str,
    task: str,
    ):
    
    if task == 'pose':
        if format == 'onnx':
            model = YOLO(f'{dir}-pose.pt')
            model.export(format=format)
            model_return = YOLO(f'{dir}-pose.onnx')
            return model
        if format == 'openvino':
            model = YOLO(f'{dir}-pose.pt')
            model.export(format=format)
            model_return = YOLO(f'{dir}-pose_openvino_model', task='pose')
            return model_return
    elif task == 'detect':
        if format == 'onnx':
            model = YOLO(f'{dir}.pt', task=task)
            model.export(format=format)
            model_return = YOLO(f'{dir}.onnx', task='detect')
            return model_return
        if format == 'openvino':
            model = YOLO(f'{dir}.pt', task='detect')
            model.export(format=format)
            model_return = YOLO(f'{dir}_openvino_model', task='detect')
            return model_return
    elif task == 'track':
        if format == 'onnx':
            model = YOLO(f'{dir}.pt', task='detect')
            model.export(format=format)
            model_return = YOLO(f'{dir}.onnx', tasl='detect')
            return YOLO(f'{dir}.onnx', tasl='detect')
        if format == 'openvino':
            model = YOLO(f'{dir}.pt', task='detect')
            model.export(format=format)
            model_return = YOLO(f'{dir}_openvino_model', task='detect')
            return model_return

def _get_model_name(
        model: str
    ):
        basename = os.path.basename(model)
        basename = basename.split('.')[0].split('-')[-0]
        if basename[-1] not in ['n', 's', 'm', 'l', 'x']:
            return basename + 'm'
        else:
            return basename
        
def _get_dir_name(
        model: str
    ):
        return os.path.dirname(model)
    
class _run_detect_model():
    """
        Using the Ultralytics YOLO model to perform object detection.
        
        Args:
            model_path (str): The path to the model file. The default is 'yolov8s.pt'.
                Models can be: yolov8(n, s, m, l, x)
            format (str): The format of the model file. It can be either 'onnx' or 'pt'.
    """
    def __init__(
        self,
        format: str,
        model_path: str = 'yolov8m.pt',
        device: str = 'cpu',
        **kwargs
    ):
        self.kwargs = kwargs
        self.format = format
        self.model_path = model_path
        self.device = device
        
        if 'iou' in kwargs:
            self.iou = kwargs['iou']
        else:
            self.iou = 0.4
        if 'conf' in kwargs:
            self.conf = kwargs['conf']
        else:
            self.conf = 0.3
        
        dir_name = _get_dir_name(model_path)
        basename = _get_model_name(model_path)
        self.basename = basename
        path = os.path.join(dir_name, basename)
        
        print(f'\n\n\nRunning Model {basename} in format {format}')
        print(f'Using IOU: {self.iou} and CONF: {self.conf}')
        print(f'Using device: {device}\n\n\n')
        
        if format == 'onnx':
            try:
                self.model = YOLO(f'{path}.onnx', task='detect')
            except FileNotFoundError:
                self.model = _create_model(dir=path, format=format, task='detect')
        elif format == 'openvino':
            try:
                self.model = YOLO(f'{path}_openvino_model', task='detect')
            except FileNotFoundError:
                self.model = _create_model(dir=path, format=format, task='detect')
        else: 
            self.model = YOLO(f'{path}.pt', task='detect')
            
    def __call__(
        self, 
        image_path: str,
    ):
        return self.predict(image_path)

    def predict(
        self, 
        image_path: str,
    ) -> list[Results]:
        
        if 'classes' in self.kwargs:
            return self.model.predict(image_path, classes=self.kwargs['classes'], device=self.device, iou=self.iou,conf=self.conf,retina_masks=True)
        
        return self.model.predict(image_path, device=self.device, iou=self.iou,conf=self.conf,retina_masks=True)


class _run_pose_model():
    """
        Using the Ultralytics YOLO model to perform object detection.
        
        Args:
            model_path (str): The path to the model file. The default is 'yolov8m-pose.pt'.
                Models can be: yolov8(n, s, m, l, x)
            format (str): The format of the model file. It can be either 'onnx' or 'pt'.
    """
    def __init__(
        self,
        model_path: str = 'yolov8m-pose.pt',
        format: str = '',
        device: str = 'cpu',
        **kwargs
    ):
        self.kwargs = kwargs
        self.format = format
        self.model_path = model_path
        self.device = device
        
        if 'iou' in kwargs:
            self.iou = kwargs['iou']
        else:
            self.iou = 0.4
        if 'conf' in kwargs:
            self.conf = kwargs['conf']
        else:
            self.conf = 0.3
        
        dir_name = _get_dir_name(model_path)
        basename = _get_model_name(model_path)
        path = os.path.join(dir_name, basename)
        
        print(f'\n\n\nRunning Model {basename} in format {format}')
        print(f'Using IOU: {self.iou} and CONF: {self.conf}')
        print(f'Using device: {device}\n\n\n')
        
        if format == 'onnx':
            try:
                self.model = YOLO(f'{path}-pose.onnx', task='pose')
            except FileNotFoundError:
                self.model = _create_model(dir=path, format=format, task='pose')
        elif format == 'openvino':
            try:
                self.model = YOLO(f'{path}-pose_openvino_model')
            except FileNotFoundError:
                self.model = _create_model(dir=path, format=format, task='pose')
        else: 
            self.model = YOLO(f'{path}-pose.pt', task='pose')
            
    def __call__(
        self, 
        image_path: str,
    ):
        return self.predict(image_path)

    def predict(
        self, 
        image_path: str,
    ) -> list[Results]:
        
        if 'classes' in self.kwargs:
            return self.model.predict(image_path, classes=self.kwargs['classes'], device=self.device, iou=self.iou,conf=self.conf,retina_masks=True)
        
        return self.model.predict(image_path, device=self.device, iou=self.iou,conf=self.conf,retina_masks=True)