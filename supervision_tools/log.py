import json
import csv
from datetime import datetime
import os
import cv2

PATH  = 'output'


class log():
    def __init__(self):
        self.log = {
            'People Count In': 0,
            'People Count Out': 0
        }
        self.frame_count = 0
        self.csv = [['tracker_id', 'frame', 'gender', 'age',  'emotion', 'position x', 'position y', 'keypoints']]
    
    def update_counter(
        self,
        count_in:int,
        count_out:int,
    ):
        self.log['People Count In'] = count_in
        self.log['People Count Out'] = count_out
        
    
    def logging(
        self, 
        object, 
        time,
        frame_count,
        result,
    ):
        """
            Log based on supervision Detections.
            Acording to Detections docs, __iter__ class defines a list of tuples with the following structure:            
                (xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
        """
        self.frame_count = frame_count
        
        # for object, time in zip(detections, times):
        xyxy = object[0]
        confidence = object[2]
        tracker_id = object[4]
        class_name = object[5]['class_name']
        
        self.log[f'Object {str(tracker_id)}'] = {
            'class_name': class_name,
            'confidence': f'{confidence*100:.0f}%',
            'time in zone': f'{time:.2f} seconds',
            'last time seen': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.populate_csv(tracker_id, xyxy, result)
            
    def populate_csv(
        self,
        tracker_id, 
        frame,
        gender,
        age,
        emotion,
        xyxy, 
        results,
    ):
        """
        CSV Header: ['tracker_id', 'frame', 'gender', 'age', 'emotion', 'position x', 'position y', 'keypoints']
        """
        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        bottom_x = int(x1 + x2 / 2)
        bottom_y = int(y2)
        
        if results is not None:
            kp_array = results[0].keypoints.xy.numpy().tolist()[0]
            int_kp_array = [[int(item) for item in sublist] for sublist in kp_array]
        else:
            int_kp_array = None
        
        self.csv.append([
            tracker_id,
            frame,
            gender,
            age,
            emotion,
            bottom_x,
            bottom_y,
            int_kp_array,
        ])
    
    def custom_logging(
        self,
        dict
    ):
        key = list(dict.keys())[0]
        self.log[key] = dict[key]
    
    def save_log(self):
        basename = 'log.json'
        csv_basename = 'log.csv'
        
        full_path = f'{PATH}/{basename}'
        
        if not os.path.exists(PATH):
            os.makedirs(PATH)
            
        with open(full_path, 'w') as file:
            json.dump(self.log, file, indent=4)
            
        with open(f'{PATH}/{csv_basename}', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.csv)
                   
class save_croped_images_in_memory():
    def __init__(
        self, 
    ):
        self.tracker_list = {}
        
    def save_image(
        self,
        object, 
        image
    ):
        self.path = PATH
        self.image = image
        self.tracker_id = object[4]
        self.xyxy = object[0]
        
        if str(self.tracker_id) not in self.tracker_list:
            self.tracker_list[str(self.tracker_id)] = str(self.xyxy)
            self.save_croped()

    def save_croped(self):
        croped_image = self.image[int(self.xyxy[1]):int(self.xyxy[3]), int(self.xyxy[0]):int(self.xyxy[2])]
        croped_image = cv2.resize(croped_image, (320, 320))
        
        # the path will be /outputs/objects_in_scene/{tracker_id}/{datetime.now().isoformat()}.jpg
        if not os.path.exists(f'{self.path}/objects_in_scene/{self.tracker_id}'):
            os.makedirs(f'{self.path}/objects_in_scene/{self.tracker_id}')
        
        basename = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        full_path = f'{self.path}/objects_in_scene/{self.tracker_id}/{basename}.jpg'
        cv2.imwrite(full_path, croped_image)